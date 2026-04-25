"""
=============================================================================
agent_orchestrator.py -- Agentic MMIX Workflow Orchestrator
=============================================================================
Uses LLM-driven decision making to run the MMIX pipeline adaptively.
The agent evaluates results at each step and decides whether to proceed,
loop back, or adjust parameters.

Architecture:
  - State machine with pipeline phases as nodes
  - LLM evaluates quality at decision points
  - Can loop back to feature engineering if model quality is poor
  - Generates reasoning trace for every decision

Usage:
    from agent_orchestrator import run_agentic_pipeline
    result = run_agentic_pipeline(granularity="weekly")
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import json
import time
import traceback
import pandas as pd

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import get_paths, get_llm_api_key, LLM_CONFIG, PipelineSummary, logger
from eda_pipeline import load_all_data, run_full_eda
from outlier_detection import run_outlier_pipeline
from data_aggregation import build_modeling_dataset
from feature_engineering import run_feature_engineering
from modeling_engine import run_modeling_pipeline
from narrative_generator import NarrativeGenerator, generate_all_narratives, call_llm, get_llm_client
from response_curves import ResponseCurveAnalyzer, run_response_curve_analysis


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState:
    """
    Tracks the full state of the agentic pipeline.
    Every decision, result, and reasoning trace is recorded here.
    """

    def __init__(self, granularity="weekly", top_n_scenarios=1, model_filter="all",
                 skip_narratives=False):
        self.granularity = granularity
        self.top_n_scenarios = top_n_scenarios
        self.model_filter = model_filter
        self.skip_narratives = skip_narratives

        # Pipeline outputs
        self.data = None
        self.clean_data = None
        self.outlier_log = None
        self.assumptions = None
        self.corr_matrix = None
        self.aggregated_data = None
        self.fe_result = None
        self.model_result = None
        self.narrator = None
        self.response_curves = None
        self.all_ranked_models = []

        # Agent tracking
        self.current_phase = "init"
        self.iteration = 0
        self.max_iterations = 3
        self.reasoning_trace = []
        self.decisions = []
        self.quality_scores = {}
        self.summary = PipelineSummary()

        # Agentic spec strategy — controls which specs are used per iteration
        # "base"     = only low-feature specs (A-H, 3 features each)
        # "expanded" = add high-feature weekly specs (K, M, O, P)
        # "groups"   = restrict to group-based specs (A, P) for VIF issues
        self.spec_strategy = "base"
        self.adjustments = {}

        # Paths
        self.paths = get_paths()
        os.makedirs(self.paths["plots_dir"], exist_ok=True)
        os.makedirs(self.paths["reports_dir"], exist_ok=True)

    def add_reasoning(self, phase, reasoning, decision, details=None):
        """Record a reasoning step."""
        entry = {
            "phase": phase,
            "iteration": self.iteration,
            "reasoning": reasoning,
            "decision": decision,
            "details": details or {},
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self.reasoning_trace.append(entry)
        self.decisions.append(decision)
        logger.info("  [AGENT] %s: %s -> %s", phase, reasoning[:80], decision)

    def get_trace_summary(self):
        """Get human-readable reasoning trace."""
        lines = ["AGENT REASONING TRACE", "=" * 60]
        for entry in self.reasoning_trace:
            lines.append(
                f"  [{entry['timestamp']}] {entry['phase']} (iter {entry['iteration']})"
            )
            lines.append(f"    Reasoning: {entry['reasoning']}")
            lines.append(f"    Decision:  {entry['decision']}")
            if entry["details"]:
                for k, v in entry["details"].items():
                    lines.append(f"    {k}: {v}")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# QUALITY EVALUATOR
# =============================================================================

class QualityEvaluator:
    """
    Evaluates pipeline outputs and decides if quality is acceptable.
    Uses both rule-based checks and LLM reasoning.
    """

    # Thresholds — aligned with LLM acceptance criteria
    # Weekly R² >= 0.50 is the bar for acceptable; below triggers RETRY
    MIN_R2 = 0.50
    MIN_ADJ_R2 = 0.45
    MAX_VIF = 50
    MIN_MODELS_PASSED = 5
    MIN_ORDINALITY_RATE = 0.5

    def __init__(self, llm_client=None):
        self.client = llm_client

    def evaluate_model_quality(self, model_result):
        """
        Evaluate modeling results. Returns quality assessment dict.

        Returns:
            {
                "acceptable": bool,
                "score": float 0-1,
                "issues": [str],
                "suggestions": [str],
                "reasoning": str,
            }
        """
        if model_result is None:
            return {
                "acceptable": False, "score": 0,
                "issues": ["Modeling returned None"],
                "suggestions": ["Check data and feature engineering"],
                "reasoning": "No models were produced.",
            }

        best = model_result.get("best_model", {})
        ranked = model_result.get("ranked_models", [])
        conv = model_result.get("convergence", {})

        issues = []
        suggestions = []

        # Check 1: Best model R2
        r2 = best.get("train_result", {}).get("r_squared", 0)
        adj_r2 = best.get("train_result", {}).get("adj_r_squared", 0)
        if r2 < self.MIN_R2:
            issues.append(f"Low R2 ({r2:.3f} < {self.MIN_R2})")
            suggestions.append("Try different feature combinations or add more features")
        if adj_r2 < self.MIN_ADJ_R2:
            issues.append(f"Low Adj R2 ({adj_r2:.3f} < {self.MIN_ADJ_R2})")

        # Check 2: Ordinality
        ord_pass = best.get("ordinality", {}).get("passed", False)
        if not ord_pass:
            issues.append("Best model fails ordinality (negative spend coefficients)")
            suggestions.append("Remove or combine collinear features")

        # Check 3: VIF
        vif = best.get("scores", {}).get("vif_max")
        if vif and vif > self.MAX_VIF:
            issues.append(f"High VIF ({vif:.0f} > {self.MAX_VIF})")
            suggestions.append("Use grouped channels instead of individual")

        # Check 4: Number of successful models
        n_success = len(ranked)
        if n_success < self.MIN_MODELS_PASSED:
            issues.append(f"Only {n_success} models succeeded (need {self.MIN_MODELS_PASSED}+)")
            suggestions.append("Check feature availability across specs")

        # Check 5: Convergence
        insights = conv.get("insights", {})
        n_confirmed = sum(1 for v in insights.values() if "confirmed" in v.get("direction", "").lower())
        n_total = len(insights)
        if n_total > 0 and n_confirmed / n_total < self.MIN_ORDINALITY_RATE:
            issues.append(f"Weak convergence ({n_confirmed}/{n_total} confirmed)")

        # Check 6: CV stability
        cv_r2 = best.get("cv_result", {}).get("cv_r2")
        if cv_r2 is not None and r2 > 0:
            stability = cv_r2 / r2
            if stability < 0.5:
                issues.append(f"Unstable model (CV R2/Train R2 = {stability:.2f})")
                suggestions.append("Reduce model complexity or add regularization")

        # Score
        score = 1.0
        score -= len(issues) * 0.15
        score = max(0, min(1, score))

        acceptable = len(issues) == 0 or (score >= 0.5 and not any("Low R2" in i for i in issues))

        reasoning = (
            f"Model quality {'ACCEPTABLE' if acceptable else 'NEEDS IMPROVEMENT'}. "
            f"Score: {score:.2f}. R2={r2:.3f}, Adj R2={adj_r2:.3f}. "
            f"{len(issues)} issues found."
        )

        return {
            "acceptable": acceptable,
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "reasoning": reasoning,
        }

    def suggest_improvements(self, quality, state, llm_verdict=None):
        """
        Based on quality issues and LLM feedback, decide what to change.
        Sets state.spec_strategy and state.adjustments so tool_run_features
        can apply the changes.

        Returns:
            dict of parameter adjustments (also stored on state)
        """
        adjustments = {}

        # Combine rule-based issues with LLM suggestions
        issues = quality.get("issues", [])

        for issue in issues:
            if "Low R2" in issue:
                if state.granularity == "monthly":
                    adjustments["granularity"] = "weekly"
                    adjustments["reason"] = "Switch to weekly for more data points"
                elif state.spec_strategy == "base":
                    adjustments["spec_strategy"] = "expanded"
                    adjustments["reason"] = "Base specs (3 features) explain too little variance. Adding high-feature specs (6-10 features) for Ridge/Lasso/ElasticNet."
                else:
                    adjustments["reason"] = "Already using expanded specs — limited further improvement possible with current data"

            elif "High VIF" in issue:
                adjustments["spec_strategy"] = "groups"
                adjustments["reason"] = "Multicollinearity detected — restricting to group-based specs"

            elif "ordinality" in issue.lower():
                adjustments["strict_ordinality"] = True
                adjustments["reason"] = "Coefficients have wrong signs — will filter to ordinality-passing models only"

            elif "Unstable" in issue:
                if state.spec_strategy == "expanded":
                    adjustments["spec_strategy"] = "base"
                    adjustments["reason"] = "High-feature specs causing instability — reverting to simpler specs"

        # If no rule-based issues but LLM said RETRY, default to expanding specs
        # This handles the case where rule-based thresholds pass but LLM
        # identifies the model as insufficient for business use.
        if not adjustments and llm_verdict and llm_verdict.get("verdict") == "RETRY":
            if state.spec_strategy == "base":
                adjustments["spec_strategy"] = "expanded"
                adjustments["reason"] = (
                    "LLM assessed model as insufficient for business use despite passing "
                    "rule-based checks. Expanding to high-feature specs to improve R²."
                )
            elif state.spec_strategy == "expanded":
                adjustments["reason"] = (
                    "LLM still unsatisfied after expanded specs. "
                    "Limited further improvement possible with current data."
                )

        # Apply strategy to state
        if "spec_strategy" in adjustments:
            state.spec_strategy = adjustments["spec_strategy"]
        state.adjustments = adjustments

        return adjustments

    def llm_evaluate(self, model_result, state):
        """
        Use LLM to reason about model quality and suggest next steps.
        Returns dict with verdict, reasoning, suggestions.
        """
        if self.client is None:
            return {"verdict": "ACCEPT", "reasoning": "LLM not available.", "suggestions": []}

        best = model_result.get("best_model", {})
        coefs = best.get("train_result", {}).get("coefficients", {})
        coef_str = "\n".join(f"  {k}: {v:+.4f}" for k, v in coefs.items() if k != "const")

        # Convergence summary
        conv = model_result.get("convergence", {}).get("insights", {})
        conv_str = "\n".join(
            f"  {f}: {i['direction']} (mean={i['mean_coefficient']:.4f}, {i['n_models']} models)"
            for f, i in sorted(conv.items(), key=lambda x: abs(x[1]["mean_coefficient"]), reverse=True)[:6]
        )

        # Quality issues from rule-based
        quality = state.quality_scores.get(state.iteration, {})
        issues_str = "\n".join(f"  - {i}" for i in quality.get("issues", [])) or "  None"

        # Build dynamic EDA context from actual data
        eda_context = ""
        if state.corr_matrix is not None:
            try:
                corr = state.corr_matrix
                target = "total_gmv"
                if target in corr.columns:
                    # NPS correlation
                    if "NPS" in corr.index:
                        nps_corr = corr.loc["NPS", target]
                        if nps_corr < -0.5:
                            eda_context += (
                                f"\n- NPS has {nps_corr:+.2f} correlation with GMV (strong negative). "
                                f"EDA determined this is a seasonality artifact — high-sale periods "
                                f"attract lower-satisfaction buyers. NPS is used as a seasonality proxy. "
                                f"A negative NPS coefficient is EXPECTED."
                            )

                    # Channel correlations - find negative ones
                    from config import MEDIA_CHANNELS
                    neg_channels = []
                    for ch in MEDIA_CHANNELS:
                        if ch in corr.index:
                            ch_corr = corr.loc[ch, target]
                            if ch_corr < -0.3:
                                neg_channels.append(f"{ch}({ch_corr:+.2f})")
                    if neg_channels:
                        eda_context += (
                            f"\n- EDA found channels with NEGATIVE GMV correlation: {', '.join(neg_channels)}. "
                            f"Negative coefficients for these channels or their groups are data-supported."
                        )
            except Exception:
                pass

        # Build convergence context
        conv_context = ""
        conv_insights = model_result.get("convergence", {}).get("insights", {})
        neg_confirmed = [f for f, i in conv_insights.items()
                        if "NEGATIVE (confirmed)" in i.get("direction", "")]
        if neg_confirmed:
            conv_context = (
                f"\n- Convergence analysis CONFIRMS these features have negative impact "
                f"across all models: {', '.join(neg_confirmed)}. "
                f"These are real findings, not model errors. Do NOT flag them as problems."
            )

        prompt = f"""You are evaluating a Marketing Mix Model for an e-commerce business.
Decide if this model is good enough for business use GIVEN THE DATA CONSTRAINTS.

DATA CONSTRAINTS:
- This is a POC with only 12 months of data (Jul 2015 - Jun 2016)
- Weekly: 47 data points. Monthly: 10-11 data points. Sample size is LIMITED.
- With small samples, R-squared of 0.70-0.80 is realistic and acceptable for weekly.
  Do NOT expect R² > 0.80 from 47 weekly data points with marketing data.
- For monthly models, R² >= 0.70 is acceptable given fewer but cleaner data points.

EDA FINDINGS (data-driven, not assumptions):
{eda_context if eda_context else '  No specific EDA context available.'}
{conv_context if conv_context else ''}

======================================================================
HARD CRITERIA (non-negotiable — if ANY hard criterion fails, verdict MUST be RETRY):
======================================================================
1. Weekly R² >= 0.50 OR Monthly R² >= 0.70. If R² is below this, verdict = RETRY. No exceptions.
2. CV MAPE < 5%. If MAPE exceeds this, verdict = RETRY.
3. Ordinality PASS. If ordinality fails (convergence-based), verdict = RETRY.
4. Sale features (sale_flag, sale_days) must have positive coefficients.

Check these FIRST. If any hard criterion fails, set verdict = "RETRY" immediately 
and explain which hard criterion failed. Do not rationalize or override.

======================================================================
SOFT CRITERIA (use judgment — these inform your reasoning but don't auto-fail):
======================================================================
- CV R² close to Train R² (stability) — large gaps are concerning but not fatal
- VIF < 10 preferred — higher is OK if model uses regularization (Ridge/Lasso)
- Coefficient magnitudes reasonable for e-commerce (elasticities typically 0.01-0.5)
- Only RETRY for genuine, FIXABLE problems — not inherent data limitations
- If all HARD criteria pass and soft criteria are mostly met, verdict should be ACCEPT

BEST MODEL:
  Spec: {best.get('spec_name', 'N/A')}
  Type: {best.get('model_type', 'N/A')}
  Transform: {best.get('transform', 'N/A')}
  R-squared: {best.get('train_result', {}).get('r_squared', 0):.4f}
  Adjusted R-squared: {best.get('train_result', {}).get('adj_r_squared', 0):.4f}
  CV R2: {best.get('cv_result', {}).get('cv_r2', 'N/A')}
  CV MAPE: {best.get('cv_result', {}).get('cv_mape', 'N/A')}%
  Ordinality: {'PASS' if best.get('ordinality', {}).get('passed') else 'FAIL'}
  VIF max: {best.get('scores', {}).get('vif_max', 'N/A')}
  Total models ranked: {len(model_result.get('ranked_models', []))}

COEFFICIENTS:
{coef_str}

INSIGHT CONVERGENCE (what all models agree on — this is the ground truth):
{conv_str}

RULE-BASED ISSUES DETECTED:
{issues_str}

CONTEXT:
  Granularity: {state.granularity}
  Data points: {state.fe_result['data'].shape[0] if state.fe_result else 'N/A'}
  Features in best model: {len(best.get('train_result', {}).get('coefficients', {})) - 1}
  Specs evaluated: {len(state.fe_result.get('feature_sets', {}))} specs with {state.spec_strategy} strategy
  Iteration: {state.iteration} of {state.max_iterations}
  Current strategy: {state.spec_strategy}

Provide your analysis in this exact JSON format:
{{
  "verdict": "ACCEPT" or "RETRY",
  "reasoning": "3-5 sentence analysis. Be realistic about what's achievable with this data size.",
  "coefficient_assessment": "Do the elasticities align with convergence and EDA findings? Flag only genuine concerns not explained by EDA.",
  "suggestions": ["specific actionable change if RETRY — not 'get more data'"]
}}"""

        response = call_llm(self.client, prompt, max_tokens=800, temperature=0.1)

        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()
            parsed = json.loads(clean)
            return parsed
        except (json.JSONDecodeError, Exception):
            return {"verdict": "ACCEPT", "reasoning": response,
                    "coefficient_assessment": "", "suggestions": []}


# =============================================================================
# PIPELINE TOOLS (callable by the agent)
# =============================================================================

def tool_load_data(state):
    """Load all datasets."""
    state.current_phase = "data_loading"
    try:
        state.data = load_all_data(state.paths["data_dir"])
        if not state.data:
            raise ValueError("No data loaded")
        state.add_reasoning("data_loading", f"Loaded {len(state.data)} datasets", "PROCEED")
        return True
    except Exception as e:
        state.add_reasoning("data_loading", f"Failed: {e}", "STOP")
        return False


def tool_run_eda(state):
    """Run EDA pipeline using already-loaded data (avoids double load)."""
    state.current_phase = "eda"
    try:
        _, classifications, issues, corr = run_full_eda(
            data_dir=state.paths["data_dir"], save_dir=state.paths["plots_dir"],
            data=state.data
        )
        state.corr_matrix = corr
        n_issues = len(issues) if issues else 0
        state.add_reasoning("eda", f"EDA complete. {n_issues} issues found.", "PROCEED")
        return True
    except Exception as e:
        state.add_reasoning("eda", f"EDA failed: {e}", "PROCEED_WITHOUT",
                           {"note": "EDA is non-critical, continuing"})
        return True  # Non-critical


def tool_run_outliers(state):
    """Run outlier detection and cleaning."""
    state.current_phase = "outlier_detection"
    try:
        state.clean_data, state.outlier_log, state.assumptions = run_outlier_pipeline(
            state.data, granularity=state.granularity, save_dir=state.paths["plots_dir"]
        )
        n_actions = len(state.outlier_log)
        state.add_reasoning("outlier_detection",
                           f"Cleaning complete. {n_actions} actions taken.",
                           "PROCEED")
        return True
    except Exception as e:
        state.add_reasoning("outlier_detection", f"Failed: {e}", "STOP")
        return False


def tool_run_aggregation(state):
    """Aggregate data to requested granularity. Caches weekly dataset to avoid recomputing."""
    state.current_phase = "aggregation"
    try:
        cache_path = os.path.join(state.paths["output_dir"], f"cached_{state.granularity}_dataset.csv")

        if os.path.exists(cache_path):
            # Load cached dataset
            cached_df = pd.read_csv(cache_path, parse_dates=["Date"] if "Date" in pd.read_csv(cache_path, nrows=1).columns else None)
            updated = state.clean_data.copy()
            updated["monthly"] = cached_df
            state.aggregated_data = updated
            n = len(cached_df)
            state.add_reasoning("aggregation",
                               f"Loaded cached {state.granularity} dataset ({n} periods) from {cache_path}.",
                               "PROCEED",
                               {"n_periods": n, "granularity": state.granularity, "cached": True})
            print(f"  [CACHE HIT] Loaded {n} periods from {cache_path}")
            return True

        # Build fresh
        result = build_modeling_dataset(state.clean_data, granularity=state.granularity)
        if result is None:
            raise ValueError("Aggregation returned None")
        state.aggregated_data = result["data"]
        n = result["n_periods"]

        # Save cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            result["data"]["monthly"].to_csv(cache_path, index=False)
            print(f"  [CACHE SAVE] Saved {state.granularity} dataset to {cache_path}")
        except Exception as ce:
            logger.warning("  Cache save failed: %s", ce)

        state.add_reasoning("aggregation",
                           f"{state.granularity} aggregation: {n} periods.",
                           "PROCEED",
                           {"n_periods": n, "granularity": state.granularity, "cached": False})
        return True
    except Exception as e:
        state.add_reasoning("aggregation", f"Failed: {e}", "STOP")
        return False


def tool_run_features(state):
    """
    Run feature engineering, then apply the agent's spec strategy.
    
    Strategy controls which specs the modeling phase will use:
      - "base":     Only low-feature specs (A-H, 3 features) — conservative start
      - "expanded": ONLY the new high-feature specs (K,M,O,P) not tested in iter 1
      - "groups":   Only group-based specs (A, P) — avoids multicollinearity
      - "monthly":  Base specs on monthly granularity (last resort)
    """
    state.current_phase = "feature_engineering"
    try:
        state.fe_result = run_feature_engineering(
            state.aggregated_data, save_dir=state.paths["plots_dir"]
        )
        if state.fe_result is None:
            raise ValueError("Feature engineering returned None")

        all_specs = state.fe_result.get("feature_sets", {})
        strategy = state.spec_strategy

        # Apply spec filtering based on agent strategy
        BASE_SPECS = {"spec_A_grouped_channels", "spec_B_total_spend", "spec_C_top_channels",
                      "spec_D_with_momentum", "spec_E_discount_effect", "spec_F_sale_duration",
                      "spec_K_all_channels"}
        WEEKLY_SPECS = {"spec_G_groups_with_discount", "spec_H_groups_with_momentum",
                        "spec_I_all_groups_full", "spec_J_groups_max_controls",
                        "spec_K_all_channels"}
        GROUP_SPECS = {"spec_A_grouped_channels", "spec_J_groups_max_controls"}

        if strategy == "base":
            filtered = {k: v for k, v in all_specs.items() if k in BASE_SPECS}
            strategy_desc = f"Base specs only ({len(filtered)} specs, max 3 features each)"
        elif strategy == "expanded":
            # Delta: only the NEW high-feature specs not tested in iteration 1
            filtered = {k: v for k, v in all_specs.items() if k in WEEKLY_SPECS}
            strategy_desc = f"Delta: {len(filtered)} new high-feature specs only (6-10 features)"
        elif strategy == "groups":
            filtered = {k: v for k, v in all_specs.items() if k in GROUP_SPECS}
            strategy_desc = f"Group-based specs only ({len(filtered)} specs, avoids channel-level VIF)"
        elif strategy == "monthly":
            filtered = {k: v for k, v in all_specs.items() if k in BASE_SPECS}
            strategy_desc = f"Monthly granularity: {len(filtered)} base specs (max 3 features for n=11)"
        else:
            filtered = all_specs
            strategy_desc = f"All {len(filtered)} specs"

        # Fallback: if filtering left too few specs, use all
        if len(filtered) < 2:
            filtered = all_specs
            strategy_desc += " (fallback: filter too aggressive, using all)"

        state.fe_result["feature_sets"] = filtered

        n_specs = len(filtered)
        n_rows = state.fe_result["data"].shape[0]
        n_cols = state.fe_result["data"].shape[1]
        state.add_reasoning("feature_engineering",
                           f"Strategy: {strategy}. {strategy_desc}. "
                           f"Matrix: {n_rows} rows × {n_cols} columns.",
                           "PROCEED",
                           {"n_specs": n_specs, "strategy": strategy,
                            "specs_used": list(filtered.keys()),
                            "shape": (n_rows, n_cols)})
        return True
    except Exception as e:
        state.add_reasoning("feature_engineering", f"Failed: {e}", "STOP")
        return False


def tool_run_modeling(state, skip_scenarios=True):
    """Run modeling pipeline. Skips scenarios during the agentic loop."""
    state.current_phase = "modeling"
    try:
        state.model_result = run_modeling_pipeline(
            state.fe_result, clean_data=state.aggregated_data,
            save_dir=state.paths["plots_dir"],
            top_n_scenarios=state.top_n_scenarios,
            model_filter=state.model_filter,
            skip_scenarios=skip_scenarios,
        )
        if state.model_result is None:
            raise ValueError("Modeling returned None")
        best = state.model_result["best_model"]
        r2 = best["train_result"]["r_squared"]
        state.all_ranked_models.extend(state.model_result["ranked_models"])
        state.add_reasoning("modeling",
                           f"Best model: {best['spec_name']} ({best['model_type']}) R2={r2:.3f}",
                           "EVALUATE",
                           {"r2": r2, "n_models": len(state.model_result["ranked_models"])})
        return True
    except Exception as e:
        state.add_reasoning("modeling", f"Failed: {e}", "STOP")
        return False


def tool_run_scenarios(state):
    """Run scenario simulation on the approved best model (post-loop)."""
    state.current_phase = "scenarios"
    try:
        from modeling_engine import (build_scenario_simulator, run_standard_scenarios,
                                     run_interactive_scenarios, plot_scenarios)
        best = state.model_result["best_model"]
        ranked = state.model_result["ranked_models"]
        fm = state.fe_result["data"]
        monthly = state.aggregated_data.get("monthly") if state.aggregated_data else None

        tc = best["transform_config"]["target"]
        fn = best["spec_config"]["resolved_features"]

        sim = build_scenario_simulator(best, fm, fn, tc, monthly)
        if sim is None:
            raise ValueError("Could not build scenario simulator")

        scenarios = run_standard_scenarios(sim)
        run_interactive_scenarios(sim)

        # Store on model_result so narratives can access them
        state.model_result["scenarios"] = scenarios
        state.model_result["simulator"] = sim

        # Plot
        plot_scenarios(scenarios, save_dir=state.paths["plots_dir"])

        state.add_reasoning("scenarios",
                           f"Ran {len(scenarios)} scenarios on approved model "
                           f"({best['spec_name']}). "
                           f"Top impact: {max(s['change_pct'] for s in scenarios):+.1f}%.",
                           "PROCEED")
        return True
    except Exception as e:
        state.add_reasoning("scenarios", f"Failed: {e}", "PROCEED_WITHOUT",
                           {"note": "Scenarios are non-critical"})
        return True


def tool_run_narratives(state):
    """Generate AI narratives with full agent context."""
    state.current_phase = "narratives"
    try:
        # Build agent context for narrator
        agent_context = {
            "iterations": state.iteration,
            "max_iterations": state.max_iterations,
            "final_strategy": state.spec_strategy,
            "quality_scores": {
                k: {"score": v.get("score"), "issues": v.get("issues", [])}
                for k, v in state.quality_scores.items()
            },
            "adjustments_made": state.adjustments,
            "decisions_summary": [
                f"Iter {e['iteration']}: {e['phase']} → {e['decision']}"
                for e in state.reasoning_trace
                if e["decision"] in ("RETRY", "PROCEED", "ISSUES_FOUND")
            ],
        }

        # Inject response curve context if available
        if state.response_curves and state.response_curves.get("narrative_context"):
            agent_context["response_curve_context"] = state.response_curves["narrative_context"]

        state.narrator = generate_all_narratives(
            {"modeling": state.model_result, "feature_engineering": state.fe_result},
            outlier_log=state.outlier_log,
            assumptions=state.assumptions,
            corr_matrix=state.corr_matrix,
            agent_context=agent_context,
        )

        state.add_reasoning("narratives", "Narratives generated with agent context.", "PROCEED")
        return True
    except Exception as e:
        state.add_reasoning("narratives", f"Failed: {e}", "PROCEED_WITHOUT",
                           {"note": "Narratives are non-critical"})
        return True


def tool_run_response_curves(state):
    """Run response curve & ROI analysis on the approved model (post-loop)."""
    state.current_phase = "response_curves"
    try:
        mr = dict(state.model_result)
        if state.all_ranked_models:
            mr["ranked_models"] = state.all_ranked_models
        rc_result = run_response_curve_analysis(
            model_result=mr,
            feature_matrix=state.fe_result["data"],
            clean_data=state.aggregated_data,
            save_dir=state.paths["plots_dir"],
        )

        # Store on state for narratives and UI
        state.response_curves = rc_result
        state.model_result["response_curves"] = rc_result

        n_channels = len(rc_result.get("channel_curves", {}))
        n_groups = len(rc_result.get("group_curves", {}))
        recs = rc_result.get("roi_summary", {}).get("recommendations", [])

        state.add_reasoning("response_curves",
                           f"Response curves computed for {n_channels} channels, "
                           f"{n_groups} groups. {len(recs)} optimization recommendations.",
                           "PROCEED",
                           {"n_channels": n_channels, "n_groups": n_groups,
                            "recommendations": recs[:3]})
        return True
    except Exception as e:
        state.add_reasoning("response_curves", f"Failed: {e}", "PROCEED_WITHOUT",
                           {"note": "Response curves are non-critical"})
        state.response_curves = None
        return True


# =============================================================================
# AGENT DECISION ENGINE
# =============================================================================

def evaluate_and_decide(state, evaluator):
    """
    Evaluate model quality and decide: proceed or loop back.
    Captures full reasoning from both rule-based and LLM evaluation.

    Returns:
        "PROCEED" or "RETRY"
    """
    print("\n" + "=" * 70)
    print("[AGENT] EVALUATING MODEL QUALITY")
    print("=" * 70)

    # Rule-based evaluation
    quality = evaluator.evaluate_model_quality(state.model_result)
    state.quality_scores[state.iteration] = quality

    print(f"\n  Rule-based assessment:")
    print(f"    Score: {quality['score']:.2f}")
    print(f"    Acceptable: {quality['acceptable']}")
    if quality["issues"]:
        print(f"    Issues:")
        for issue in quality["issues"]:
            print(f"      - {issue}")
    if quality["suggestions"]:
        print(f"    Suggestions:")
        for s in quality["suggestions"]:
            print(f"      - {s}")

    # Record rule-based reasoning
    state.add_reasoning("rule_based_evaluation",
                       quality["reasoning"],
                       "ACCEPTABLE" if quality["acceptable"] else "ISSUES_FOUND",
                       {"score": quality["score"],
                        "issues": quality["issues"],
                        "suggestions": quality["suggestions"]})

    # LLM evaluation
    llm_verdict = None
    if evaluator.client:
        print(f"\n  LLM evaluation...")
        try:
            llm_verdict = evaluator.llm_evaluate(state.model_result, state)
            state.last_llm_verdict = llm_verdict  # Store for suggest_improvements
            print(f"    Verdict: {llm_verdict.get('verdict', 'N/A')}")
            print(f"    Reasoning: {llm_verdict.get('reasoning', 'N/A')}")
            print(f"    Coefficient Assessment: {llm_verdict.get('coefficient_assessment', 'N/A')}")
            if llm_verdict.get("suggestions"):
                print(f"    Suggestions:")
                for s in llm_verdict["suggestions"]:
                    print(f"      - {s}")

            # Record full LLM reasoning
            state.add_reasoning("llm_evaluation",
                               llm_verdict.get("reasoning", "No reasoning provided"),
                               llm_verdict.get("verdict", "N/A"),
                               {"coefficient_assessment": llm_verdict.get("coefficient_assessment", ""),
                                "suggestions": llm_verdict.get("suggestions", [])})
        except Exception as e:
            logger.error("  LLM evaluation failed: %s", e)

    # Final decision — LLM verdict carries significant weight.
    # If LLM says RETRY and we still have iterations left, we retry.
    # This makes the agent truly adaptive rather than just threshold-based.
    if state.iteration >= state.max_iterations:
        decision = "PROCEED"
        reasoning = (f"Max iterations ({state.max_iterations}) reached. "
                    f"Proceeding with best available results. "
                    f"Final quality score: {quality['score']:.2f}.")
    elif llm_verdict and llm_verdict.get("verdict") == "RETRY":
        # LLM says RETRY — respect it regardless of rule-based outcome
        decision = "RETRY"
        reasoning = (f"LLM recommends RETRY. "
                    f"Rule-based score={quality['score']:.2f} "
                    f"({'acceptable' if quality['acceptable'] else 'issues found'}). "
                    f"LLM reasoning: {llm_verdict.get('reasoning', '')[:200]}. "
                    f"Will retry with adjusted strategy.")
    elif not quality["acceptable"]:
        # Rule-based flagged issues, no LLM override
        decision = "RETRY"
        reasoning = (f"Quality score {quality['score']:.2f} below threshold. "
                    f"Issues: {', '.join(quality['issues'][:3])}. "
                    f"Will retry with adjustments.")
    else:
        decision = "PROCEED"
        reasoning = (f"Quality acceptable. Score={quality['score']:.2f}. "
                    f"R2={state.model_result['best_model']['train_result']['r_squared']:.3f}. "
                    f"LLM also approved.")

    state.add_reasoning("final_decision", reasoning, decision)

    return decision


# =============================================================================
# MAIN AGENTIC PIPELINE
# =============================================================================

def run_agentic_pipeline(granularity="weekly", top_n_scenarios=1,
                         model_filter="all", skip_eda=False,
                         skip_narratives=False,
                         data_dir=None, output_dir=None):
    """
    Run the MMIX pipeline with agentic decision-making.

    Flow:
      Phase 1-4: Data Loading → EDA → Cleaning → Aggregation (run once)
      Agentic Loop (max 3 iterations):
        Phase 5: Feature Engineering (specs filtered by agent strategy)
        Phase 6: Modeling (scenarios SKIPPED during loop)
        Phase 7: Quality Evaluation (rule-based + LLM)
        → RETRY: agent adjusts spec_strategy, loops back to Phase 5
        → PROCEED: exit loop
      Phase 8: Scenario Simulation (on approved model only)
      Phase 9: Narrative Generation (with full agent context)

    Iteration Strategy:
      Iter 1: "base" — only low-feature specs (A-H, 3 features)
      If RETRY with Low R²: "expanded" — add high-feature specs (K,M,O,P)
      If RETRY with High VIF: "groups" — restrict to group-based specs
      If RETRY with Unstable: revert to "base" simpler specs

    Args:
        granularity:     'weekly' or 'monthly'
        top_n_scenarios: models for scenarios
        model_filter:    'all', 'linear', or comma-separated
        skip_eda:        skip EDA
        skip_narratives: skip AI narrative generation
        data_dir:        data folder override
        output_dir:      output folder override

    Returns:
        AgentState with all results and reasoning trace
    """
    start_time = time.time()

    # Initialize
    state = AgentState(granularity, top_n_scenarios, model_filter, skip_narratives)
    if data_dir or output_dir:
        state.paths = get_paths(data_dir, output_dir)

    # LLM client for evaluation
    llm_client = get_llm_client()
    evaluator = QualityEvaluator(llm_client)

    print("=" * 70)
    print("  AGENTIC MMIX PIPELINE")
    print(f"  Granularity: {granularity} | Models: {model_filter}")
    print(f"  Max iterations: {state.max_iterations}")
    print(f"  Strategy: Start with base specs → expand if needed")
    print("=" * 70)

    state.add_reasoning("init",
                       f"Starting agentic pipeline: {granularity}, models={model_filter}. "
                       f"Iteration 1 will use base specs (3 features). "
                       f"Agent will expand to high-feature specs if R² is insufficient.",
                       "PROCEED")

    # =====================================================================
    # PHASE 1: Data Loading (run once)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 1] DATA LOADING")
    print("=" * 70)
    if not tool_load_data(state):
        return state

    # =====================================================================
    # PHASE 2: EDA (run once, optional — uses pre-loaded data)
    # =====================================================================
    if not skip_eda:
        print("\n" + "=" * 70)
        print("[PHASE 2] EDA")
        print("=" * 70)
        tool_run_eda(state)
    else:
        state.add_reasoning("eda", "Skipped by user", "SKIP")

    # =====================================================================
    # PHASE 3: Outlier Detection (run once)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 3] OUTLIER DETECTION")
    print("=" * 70)
    if not tool_run_outliers(state):
        return state

    # =====================================================================
    # PHASE 4: Aggregation (run once)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 4] AGGREGATION")
    print("=" * 70)
    if not tool_run_aggregation(state):
        return state

    # =====================================================================
    # AGENTIC LOOP: Features → Modeling → Evaluate → (Retry?)
    # Scenarios are NOT run inside this loop.
    #
    # Iteration strategy:
    #   Iter 1: "base" specs (A-H, 3 features each)
    #   Iter 2: "expanded" — ONLY the new high-feature specs (K,M,O,P)
    #           that weren't in iter 1. Compare best vs iter 1 best.
    #   Iter 3: "monthly" — switch granularity to monthly as last resort
    # =====================================================================
    prev_best = None  # Track best model from previous iteration

    while state.iteration < state.max_iterations:
        state.iteration += 1

        print(f"\n{'='*70}")
        print(f"  ITERATION {state.iteration} of {state.max_iterations}")
        print(f"  Spec strategy: {state.spec_strategy}")
        print(f"{'='*70}")

        # PHASE 5: Feature Engineering (specs filtered by strategy)
        print(f"\n[PHASE 5] FEATURE ENGINEERING (iter {state.iteration}, strategy={state.spec_strategy})")
        if not tool_run_features(state):
            break

        # PHASE 6: Modeling (scenarios skipped — will run after loop)
        print(f"\n[PHASE 6] MODELING (iter {state.iteration}, no scenarios)")
        if not tool_run_modeling(state, skip_scenarios=True):
            break

        # If we have a previous best, compare and keep the winner
        current_best = state.model_result["best_model"]
        current_r2 = current_best["train_result"]["r_squared"]

        if prev_best is not None:
            prev_r2 = prev_best["train_result"]["r_squared"]
            if current_r2 > prev_r2:
                state.add_reasoning("model_comparison",
                    f"Iter {state.iteration} best R2={current_r2:.3f} > "
                    f"prev best R2={prev_r2:.3f}. Keeping new model.",
                    "IMPROVED",
                    {"prev_r2": prev_r2, "new_r2": current_r2,
                     "improvement": f"+{(current_r2 - prev_r2):.3f}"})
            else:
                # Previous was better — restore it
                state.add_reasoning("model_comparison",
                    f"Iter {state.iteration} best R2={current_r2:.3f} <= "
                    f"prev best R2={prev_r2:.3f}. Keeping previous model.",
                    "NO_IMPROVEMENT",
                    {"prev_r2": prev_r2, "new_r2": current_r2})
                state.model_result["best_model"] = prev_best

        # PHASE 7: Evaluate
        decision = evaluate_and_decide(state, evaluator)

        if decision == "PROCEED":
            break
        elif decision == "RETRY":
            # Save current best before retry
            prev_best = state.model_result["best_model"]

            # Agent determines adjustments and updates spec_strategy
            quality = state.quality_scores[state.iteration]
            llm_v = getattr(state, 'last_llm_verdict', None)
            adjustments = evaluator.suggest_improvements(quality, state, llm_verdict=llm_v)

            # Iteration 3 fallback: try monthly granularity
            if state.iteration >= 2 and state.spec_strategy == "expanded":
                state.spec_strategy = "monthly"
                adjustments["spec_strategy"] = "monthly"
                adjustments["reason"] = (
                    "Expanded weekly specs did not satisfy quality requirements. "
                    "Switching to monthly granularity as final attempt — "
                    "fewer data points but aggregates away weekly noise."
                )

            state.add_reasoning("retry",
                               f"Agent adjusting strategy: {state.spec_strategy}. "
                               f"Reason: {adjustments.get('reason', 'N/A')}.",
                               "RETRY",
                               {"adjustments": adjustments,
                                "new_strategy": state.spec_strategy,
                                "issues": quality.get("issues", [])})
            print(f"\n  [AGENT] Strategy changed to: {state.spec_strategy}")
            print(f"  [AGENT] Reason: {adjustments.get('reason', 'N/A')}")

            # If switching to monthly, re-aggregate
            if state.spec_strategy == "monthly" and state.granularity != "monthly":
                print(f"\n  [AGENT] Re-aggregating data for monthly granularity...")
                state.granularity = "monthly"
                if not tool_run_aggregation(state):
                    break

    # =====================================================================
    # PHASE 8: Scenario Simulation (on approved model only)
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 8] SCENARIO SIMULATION (on approved model)")
    print("=" * 70)
    tool_run_scenarios(state)

    # =====================================================================
    # PHASE 8.5: Response Curves & ROI Analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 8.5] RESPONSE CURVES & ROI OPTIMIZATION")
    print("=" * 70)
    tool_run_response_curves(state)

    # =====================================================================
    # PHASE 9: Narratives (with full agent context)
    # =====================================================================
    if state.skip_narratives:
        print("\n" + "=" * 70)
        print("[PHASE 9] NARRATIVES (SKIPPED by user)")
        print("=" * 70)
        state.add_reasoning("narratives", "Skipped by user request (--skip-narratives)", "SKIP")
    else:
        print("\n" + "=" * 70)
        print("[PHASE 9] NARRATIVES (with agent context)")
        print("=" * 70)
        tool_run_narratives(state)

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("[DONE] AGENTIC PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Iterations:      {state.iteration}")
    print(f"  Final strategy:  {state.spec_strategy}")
    print(f"  Decisions:       {len(state.decisions)}")
    print(f"  Final quality:   {state.quality_scores.get(state.iteration, {}).get('score', 'N/A')}")
    print(f"  Runtime:         {elapsed:.1f}s")

    # Quality progression
    if len(state.quality_scores) > 1:
        print(f"\n  Quality progression:")
        for it, qs in sorted(state.quality_scores.items()):
            print(f"    Iter {it}: score={qs.get('score', 'N/A'):.2f}, "
                  f"issues={qs.get('issues', [])}")

    # Print reasoning trace
    print(f"\n{state.get_trace_summary()}")

    # Save reasoning trace
    trace_path = os.path.join(state.paths["reports_dir"], "agent_reasoning_trace.txt")
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(state.get_trace_summary())
        print(f"  Trace saved to: {trace_path}")
    except Exception as e:
        logger.error("  Failed to save trace: %s", e)

    return state


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic MMIX Pipeline")
    parser.add_argument("-g", "--granularity", choices=["weekly", "monthly"],
                       default="weekly", help="Time granularity")
    parser.add_argument("-t", "--top-models", type=int, default=1,
                       help="Top models for scenarios")
    parser.add_argument("-m", "--models", type=str, default="all",
                       help="Model filter: all, linear, or comma-separated")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    state = run_agentic_pipeline(
        granularity=args.granularity,
        top_n_scenarios=args.top_models,
        model_filter=args.models,
        skip_eda=args.skip_eda,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )