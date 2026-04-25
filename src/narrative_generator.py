"""
=============================================================================
narrative_generator.py -- GenAI Business Narratives
=============================================================================
Generates business-ready explanations at every pipeline step using GPT-4.1.
Reads structured summaries from pipeline modules and produces plain-language
narratives for marketing stakeholders.

Usage:
    from narrative_generator import NarrativeGenerator
    narrator = NarrativeGenerator()
    narrative = narrator.narrate_eda(eda_summary)
    narrative = narrator.narrate_full_pipeline(pipeline_summary)
=============================================================================
"""

import os
import json
import time
from config import LLM_CONFIG, MEDIA_CHANNELS, get_llm_api_key, logger

# =============================================================================
# LLM CLIENT
# =============================================================================

def get_llm_client():
    """Initialize Azure OpenAI client."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None

    key = get_llm_api_key()
    if not key:
        logger.error("API key not found. Set %s in .env", LLM_CONFIG["api_key_env"])
        return None

    try:
        client = AzureOpenAI(
            azure_endpoint=LLM_CONFIG["endpoint"],
            api_key=key,
            api_version=LLM_CONFIG["api_version"],
        )
        logger.info("  [OK] Azure OpenAI client initialized")
        return client
    except Exception as e:
        logger.error("  Failed to create LLM client: %s", e)
        return None


def call_llm(client, prompt, system_prompt=None, max_tokens=1500, temperature=0.3):
    """
    Call GPT-4.1 and return response text.

    Args:
        client:        AzureOpenAI client
        prompt:        user message
        system_prompt: system context (defaults to MMIX analyst role)
        max_tokens:    response length limit
        temperature:   0.0-1.0 (lower = more focused)

    Returns:
        str response or error message
    """
    if client is None:
        return "[Narrative unavailable -- LLM client not initialized]"

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    try:
        response = client.chat.completions.create(
            model=LLM_CONFIG["deployment"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("  LLM call failed: %s", e)
        return f"[Narrative generation failed: {e}]"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a senior marketing analytics consultant specializing in 
Marketing Mix Modeling (MMIX) for e-commerce. You explain data findings in clear, 
actionable business language for marketing directors and C-suite executives.

Rules:
- Be concise but insightful (3-5 sentences per section unless more detail is needed)
- Always structure as: What happened -> Why it matters -> What to do about it
- Use business language, not statistical jargon
- When mentioning numbers, round to 1 decimal place and use Cr (Crores) for currency
- Highlight actionable recommendations
- Flag risks and limitations honestly
- Do not make up numbers -- only use what is provided in the data"""


# =============================================================================
# NARRATIVE GENERATOR CLASS
# =============================================================================

class NarrativeGenerator:
    """
    Generates business narratives for each pipeline step.
    Initialize once, call methods per step.
    """

    def __init__(self):
        """Initialize LLM client."""
        logger.info("Initializing Narrative Generator...")
        self.client = get_llm_client()
        self.narratives = {}

    def is_available(self):
        """Check if LLM is available."""
        return self.client is not None

    # -----------------------------------------------------------------
    # STEP-LEVEL NARRATIVES
    # -----------------------------------------------------------------

    def narrate_eda(self, data_summary, corr_summary=None, sale_lift=None):
        """
        Generate narrative for EDA findings.

        Args:
            data_summary:  str describing datasets loaded and basic stats
            corr_summary:  str of channel correlations with GMV
            sale_lift:     float, percentage lift during sales
        """
        prompt = f"""Analyze these EDA findings from an e-commerce Marketing Mix dataset 
and write a business-ready summary.

DATA OVERVIEW:
{data_summary}

CHANNEL CORRELATIONS WITH REVENUE (GMV):
{corr_summary or 'Not available'}

SALE EVENT IMPACT:
{f'Sale days show {sale_lift:.1f}% lift in GMV compared to non-sale days' if sale_lift else 'Not calculated'}

Write a 2-3 paragraph executive summary covering:
1. What the data tells us about revenue drivers
2. Which marketing channels appear most effective
3. Key risks or data quality concerns to be aware of"""

        narrative = call_llm(self.client, prompt)
        self.narratives["eda"] = narrative
        return narrative

    def narrate_outliers(self, cleaning_log, assumptions):
        """
        Generate narrative for outlier detection decisions.

        Args:
            cleaning_log:  list of cleaning action dicts
            assumptions:   dict of documented assumptions
        """
        log_text = "\n".join(
            f"- {entry.get('reason', str(entry))}" for entry in cleaning_log
        )
        assumption_text = "\n".join(
            f"- {key}: {a['decision']} (Reasoning: {a['reasoning'][:100]}...)"
            for key, a in assumptions.items()
        )

        prompt = f"""Explain these data cleaning decisions for a marketing stakeholder.

CLEANING ACTIONS TAKEN:
{log_text}

KEY ASSUMPTIONS:
{assumption_text}

Write a clear explanation covering:
1. What was cleaned and why
2. What was intentionally kept (and why keeping it is the right choice)
3. Any risks from these decisions"""

        narrative = call_llm(self.client, prompt)
        self.narratives["outliers"] = narrative
        return narrative

    def narrate_features(self, summaries, vif_summary=None, channel_ranking=None):
        """
        Generate narrative for feature engineering.

        Args:
            summaries:       dict of {step_name: summary_text} from feature engineering
            vif_summary:     str summarizing multicollinearity findings
            channel_ranking: str showing data-driven channel ranking by correlation
        """
        steps_text = "\n".join(
            f"- {name}: {summary}" for name, summary in summaries.items()
        )

        multicollinearity_context = ""
        if vif_summary:
            multicollinearity_context = f"""
MULTICOLLINEARITY ANALYSIS:
{vif_summary}
"""
        if channel_ranking:
            multicollinearity_context += f"""
CHANNEL RANKING (by correlation with GMV):
{channel_ranking}
Note: Some channels are highly correlated with each other (e.g. Online.marketing 
and Affiliates r=0.99), which is why some specs use channel groups instead of 
individual channels. Ridge/Lasso/ElasticNet models handle this multicollinearity 
via regularization, so high-feature-count specs (7-9 features) are included 
specifically for these regularized model types.
"""

        prompt = f"""Explain these feature engineering steps for a marketing director 
who needs to understand what inputs the model uses.

FEATURE ENGINEERING STEPS:
{steps_text}
{multicollinearity_context}
Write a clear explanation covering:
1. What features were created and why each matters for understanding marketing ROI
2. How channels were grouped and the business logic behind it
3. Why some model specs use individual channels while others use groups (multicollinearity)
4. Any transformations applied and their business interpretation
5. Limitations of the feature set"""

        narrative = call_llm(self.client, prompt)
        self.narratives["features"] = narrative
        return narrative

    def narrate_modeling(self, best_model, top10, convergence, vif_info=None, agent_context=None):
        """
        Generate narrative for modeling results.

        Args:
            best_model:     dict with best model details
            top10:          list of top 10 model dicts
            convergence:    dict with insight convergence analysis
            vif_info:       str summarizing VIF for best model (optional)
            agent_context:  dict with iteration info from agentic pipeline (optional)
        """
        # Best model summary
        best = best_model
        coef_text = "\n".join(
            f"  {f}: {c:+.4f}" for f, c in best["train_result"]["coefficients"].items()
            if f != "const"
        )

        # Top 10 summary
        top_text = "\n".join(
            f"  #{r['rank']} {r['spec_name']} ({r['model_type']}, {r['transform']}) "
            f"-- Composite: {r['scores']['composite']:.3f}, R2: {r['train_result']['r_squared']:.3f}"
            for r in top10[:5]
        )

        # Convergence
        conv_text = "\n".join(
            f"  {f}: {i['direction']} (mean={i['mean_coefficient']:.4f}, {i['n_models']} models)"
            for f, i in sorted(convergence.get("insights", {}).items(),
                                key=lambda x: abs(x[1]["mean_coefficient"]), reverse=True)[:8]
        )

        vif_context = ""
        if vif_info:
            vif_context = f"""
MULTICOLLINEARITY (VIF) FOR BEST MODEL:
{vif_info}
Note: Models were scored on a composite metric that includes a 15% weight for VIF.
High-feature specs (7-9 channels) are handled by Ridge/Lasso/ElasticNet which use 
regularization to produce stable coefficients despite correlated inputs.
"""

        agent_context_text = ""
        if agent_context:
            n_iters = agent_context.get("iterations", 1)
            if n_iters > 1:
                quality_prog = ""
                for it, qs in sorted(agent_context.get("quality_scores", {}).items()):
                    quality_prog += f"\n  Iteration {it}: score={qs.get('score', 'N/A')}, issues={qs.get('issues', [])}"
                agent_context_text = f"""
AGENT ITERATION HISTORY:
  The agentic pipeline ran {n_iters} iterations before accepting this model.
  Final strategy: {agent_context.get('final_strategy', 'N/A')}
  Quality progression:{quality_prog}
  Adjustments made: {agent_context.get('adjustments_made', 'None')}
  This means the agent detected quality issues in earlier iterations and
  adjusted its feature selection strategy to improve results.
"""

            # Always include response curve analysis if available
            rc_context = agent_context.get("response_curve_context")
            if rc_context:
                agent_context_text += f"\n{rc_context}\n"

        prompt = f"""Interpret these Marketing Mix Model results for a business audience.

BEST MODEL:
  Spec: {best['spec_name']}
  Type: {best['model_type']} ({best['transform']})
  R-squared: {best['train_result']['r_squared']:.4f}
  Adjusted R-squared: {best['train_result']['adj_r_squared']:.4f}
  CV MAPE: {best['cv_result'].get('cv_mape', 'N/A')}%
  Coefficients (elasticities):
{coef_text}

TOP 5 MODELS:
{top_text}

INSIGHT CONVERGENCE (what all models agree on):
{conv_text}
{vif_context}
MODEL SELECTION NOTE:
Only linear models (OLS, Ridge, Lasso, ElasticNet, Bayesian Ridge, Huber) were used.
Tree-based models (XGBoost, Random Forest) were excluded because they do not produce
interpretable elasticities -- which are essential for marketing budget recommendations.
{agent_context_text}
Write a business interpretation covering:
1. What the best model tells us about marketing effectiveness
2. Which channels drive the most revenue impact (interpret the elasticities)
3. How confident we should be in these findings (reference R2, CV, convergence)
4. Key limitations and caveats"""

        narrative = call_llm(self.client, prompt)
        self.narratives["modeling"] = narrative
        return narrative

    def narrate_scenarios(self, scenario_results, response_curve_context=None):
        """Generate narrative grounded in ROI data, not just scenario percentages."""
        scen_text = "\n".join(
            f"  {s['scenario_name']}: {s['change_pct']:+.1f}% GMV change "
            f"(Baseline: {s['baseline_gmv']/1e7:.2f} Cr -> {s['predicted_gmv']/1e7:.2f} Cr)"
            for s in scenario_results
        )

        if response_curve_context:
            prompt = f"""You are writing budget recommendations for a marketing director.

PRIMARY DATA — USE THIS FOR ALL RECOMMENDATIONS:
{response_curve_context}

SUPPORTING DATA — scenario simulations:
{scen_text}

CRITICAL INSTRUCTION: Base every recommendation on the marginal ROI numbers and
optimal spend points above. Do NOT just say "+20% in channel X gives Y% GMV lift".
Instead say "Channel X returns ₹Z per ₹1 at current spend, with optimal spend at
N× current level" and recommend specific ₹ reallocation amounts.

Write recommendations covering:
1. Overall media ROI — total return on media investment
2. Per-channel ROI ranking — which channels to increase/decrease with specific ₹ marginal returns
3. Optimal spend levels — where each channel sits vs its saturation point
4. Budget reallocation — specific shifts with expected ₹ impact
5. Sale events vs media spend — which lever is stronger (use scenario data here)
6. Recommended strategy with ₹ amounts
7. Caveats (per-channel estimates are directional if from secondary model)"""
        else:
            prompt = f"""Translate these marketing budget scenario simulations into
actionable recommendations for a marketing director.

SCENARIO RESULTS:
{scen_text}

Write specific recommendations covering:
1. Which channels to increase/decrease
2. Budget reallocation opportunities
3. Sale events vs media spend
4. Recommended strategy
5. Risks"""

        narrative = call_llm(self.client, prompt)
        self.narratives["scenarios"] = narrative
        return narrative

    # -----------------------------------------------------------------
    # FULL PIPELINE NARRATIVE
    # -----------------------------------------------------------------

    def narrate_full_pipeline(self, pipeline_summary_text):
        """
        Generate a complete executive report from the pipeline summary.

        Args:
            pipeline_summary_text: str from PipelineSummary.get_full_summary()
        """
        prompt = f"""Write a complete executive report from this Marketing Mix analysis.

PIPELINE SUMMARY:
{pipeline_summary_text}

Structure the report as:
1. EXECUTIVE SUMMARY (3-4 sentences)
2. KEY FINDINGS (what drives revenue)
3. CHANNEL EFFECTIVENESS (rank channels by impact)
4. RECOMMENDATIONS (specific budget actions)
5. RISKS & LIMITATIONS (what we don't know)
6. NEXT STEPS (what to do with more data/time)

Keep it under 500 words. Use business language, not technical jargon."""

        narrative = call_llm(self.client, prompt, max_tokens=2000)
        self.narratives["full_report"] = narrative
        return narrative

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    def get_all_narratives(self):
        """Return all generated narratives as a dict."""
        return self.narratives

    def save_report(self, output_dir=None):
        """Save all narratives to a markdown file."""
        if not self.narratives:
            logger.warning("No narratives to save")
            return None

        if output_dir is None:
            from config import get_paths
            output_dir = get_paths()["reports_dir"]
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, "mmix_narrative_report.md")

        sections = {
            "eda": "Exploratory Data Analysis",
            "outliers": "Data Quality & Cleaning",
            "features": "Feature Engineering",
            "modeling": "Model Results",
            "scenarios": "Scenario Recommendations",
            "full_report": "Executive Report",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Marketing Mix Analysis -- AI-Generated Report\n\n")
            f.write(f"*Generated automatically by the Agentic MMIX Pipeline*\n\n")
            f.write("---\n\n")

            for key, title in sections.items():
                if key in self.narratives:
                    f.write(f"## {title}\n\n")
                    f.write(self.narratives[key])
                    f.write("\n\n---\n\n")

        logger.info("  [OK] Report saved to %s", filepath)
        return filepath

    def print_all(self):
        """Print all narratives to console."""
        sections = {
            "eda": "EXPLORATORY DATA ANALYSIS",
            "outliers": "DATA QUALITY & CLEANING",
            "features": "FEATURE ENGINEERING",
            "modeling": "MODEL RESULTS",
            "scenarios": "SCENARIO RECOMMENDATIONS",
            "full_report": "EXECUTIVE REPORT",
        }
        for key, title in sections.items():
            if key in self.narratives:
                print(f"\n{'='*70}")
                print(f"  {title}")
                print(f"{'='*70}")
                print(self.narratives[key])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_all_narratives(pipeline_result, outlier_log=None, assumptions=None,
                            corr_matrix=None, save=True, agent_context=None):
    """
    One-call function to generate all narratives from pipeline output.

    Args:
        pipeline_result: output from run_pipeline() in main.py
        outlier_log:     list of cleaning actions
        assumptions:     assumptions dict
        corr_matrix:     correlation matrix from EDA
        save:            whether to save report to file

    Returns:
        NarrativeGenerator instance with all narratives
    """
    print("\n" + "=" * 70)
    print("[START] GENERATING AI NARRATIVES")
    print("=" * 70)

    narrator = NarrativeGenerator()
    if not narrator.is_available():
        logger.error("LLM not available -- skipping narratives")
        return narrator

    # Determine structure based on granularity
    if isinstance(pipeline_result, dict) and "modeling" in pipeline_result:
        # Single granularity result
        modeling = pipeline_result["modeling"]
        fe = pipeline_result.get("feature_engineering", {})
    elif isinstance(pipeline_result, dict) and "weekly" in pipeline_result:
        # Both granularities -- use weekly
        modeling = pipeline_result["weekly"]["modeling"]
        fe = pipeline_result["weekly"].get("feature_engineering", {})
    else:
        logger.error("Unexpected pipeline result structure")
        return narrator

    # 1. EDA narrative
    print("\n[1/5] Generating EDA narrative...")
    try:
        data_summary = (
            f"E-commerce dataset with 12 months of data (Jul 2015 - Jun 2016). "
            f"5 product categories: Camera, CameraAccessory, EntertainmentSmall, "
            f"GameCDDVD, GamingHardware. "
            f"9 media channels: TV, Digital, Sponsorship, Content Marketing, "
            f"Online marketing, Affiliates, SEM, Radio, Other. "
            f"Monthly GMV ranges from ~18 Cr to ~51 Cr."
        )
        corr_text = None
        if corr_matrix is not None:
            try:
                ch_corr = corr_matrix.loc[
                    [c for c in corr_matrix.index if c in
                     ["TV", "Digital", "Sponsorship", "Content.Marketing",
                      "Online.marketing", "Affiliates", "SEM", "Radio", "Other"]],
                    "total_gmv"
                ].sort_values(ascending=False)
                corr_text = "\n".join(
                    f"  {ch}: {val:+.3f}" for ch, val in ch_corr.items()
                )
            except Exception:
                pass
        narrator.narrate_eda(data_summary, corr_text, sale_lift=116.5)
        print("  [OK] EDA narrative generated")
    except Exception as e:
        logger.error("  EDA narrative failed: %s", e)

    # 2. Outlier narrative
    print("\n[2/5] Generating outlier narrative...")
    try:
        if outlier_log and assumptions:
            narrator.narrate_outliers(outlier_log, assumptions)
            print("  [OK] Outlier narrative generated")
        else:
            logger.info("  [SKIP] No outlier log/assumptions provided")
    except Exception as e:
        logger.error("  Outlier narrative failed: %s", e)

    # 3. Feature engineering narrative
    print("\n[3/5] Generating feature engineering narrative...")
    try:
        fe_summaries = fe.get("summaries", {}) if isinstance(fe, dict) else {}
        if fe_summaries:
            # Build multicollinearity context from VIF results
            vif_summary = None
            vif_results = fe.get("vif_results", {}) if isinstance(fe, dict) else {}
            if vif_results:
                vif_lines = []
                for spec_name, vr in vif_results.items():
                    if vr.get("vif") is not None:
                        max_vif = vr["vif"]["VIF"].max()
                        high_count = (vr["vif"]["VIF"] > 10).sum()
                        vif_lines.append(f"  {spec_name}: max VIF={max_vif:.1f}, {high_count} features >10")
                if vif_lines:
                    vif_summary = "\n".join(vif_lines)

            # Build channel ranking context from correlation matrix
            channel_ranking = None
            if corr_matrix is not None:
                try:
                    ch_cols = [c for c in corr_matrix.index if c in MEDIA_CHANNELS]
                    if ch_cols and "total_gmv" in corr_matrix.columns:
                        ch_corr = corr_matrix.loc[ch_cols, "total_gmv"].sort_values(ascending=False)
                        channel_ranking = "\n".join(
                            f"  {ch}: {val:+.3f}" for ch, val in ch_corr.items()
                        )
                except Exception:
                    pass

            narrator.narrate_features(fe_summaries, vif_summary=vif_summary, channel_ranking=channel_ranking)
            print("  [OK] Feature narrative generated")
        else:
            logger.info("  [SKIP] No feature summaries available")
    except Exception as e:
        logger.error("  Feature narrative failed: %s", e)

    # 4. Modeling narrative
    print("\n[4/5] Generating modeling narrative...")
    try:
        if modeling:
            # Build VIF info for the best model
            best_vif_info = None
            best_spec = modeling["best_model"].get("spec_name", "")
            if isinstance(fe, dict):
                vif_results = fe.get("vif_results", {})
                best_vif = vif_results.get(best_spec, {})
                if best_vif and best_vif.get("vif") is not None:
                    vif_df = best_vif["vif"]
                    best_vif_info = "\n".join(
                        f"  {row['feature']}: VIF={row['VIF']:.1f}"
                        for _, row in vif_df.iterrows()
                    )

            narrator.narrate_modeling(
                modeling["best_model"],
                modeling["top_10"],
                modeling["convergence"],
                vif_info=best_vif_info,
                agent_context=agent_context,
            )
            print("  [OK] Modeling narrative generated")
    except Exception as e:
        logger.error("  Modeling narrative failed: %s", e)

    # 5. Scenario narrative
    print("\n[5/5] Generating scenario narrative...")
    try:
        scenarios = modeling.get("scenarios", []) if modeling else []
        if scenarios:
            rc_context = agent_context.get("response_curve_context") if agent_context else None
            narrator.narrate_scenarios(scenarios, response_curve_context=rc_context)
            print("  [OK] Scenario narrative generated")
    except Exception as e:
        logger.error("  Scenario narrative failed: %s", e)

    # Print all
    narrator.print_all()

    # Save
    if save:
        filepath = narrator.save_report()
        if filepath:
            print(f"\n  Report saved to: {filepath}")

    print("\n" + "=" * 70)
    print("[DONE] NARRATIVES COMPLETE")
    print("=" * 70)

    return narrator


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    # Quick test -- just check LLM connection
    narrator = NarrativeGenerator()
    if narrator.is_available():
        response = call_llm(narrator.client, "Say 'Connection successful' in one sentence.")
        print(f"\nLLM Test: {response}")
    else:
        print("\nLLM not available. Check .env file and API key.")