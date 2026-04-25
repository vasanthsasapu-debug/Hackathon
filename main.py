"""
=============================================================================
MAIN RUNNER -- Agentic E-Commerce MMIX Pipeline
=============================================================================
All runs go through the agentic pipeline. If the model passes quality
checks on the first iteration, it proceeds immediately (zero overhead).
If not, the agent iterates with adjusted feature strategies.

Usage:
    python main.py                                    # Default: weekly
    python main.py -g weekly -t 3                     # Weekly, top 3 scenarios
    python main.py -g monthly                         # Monthly
    python main.py -g both                            # Both with comparison
    python main.py --skip-eda                         # Skip EDA
    python main.py --skip-narratives                  # Skip AI narratives
    python main.py -m linear                          # Linear models only
    python main.py --data-dir /path/to/data           # Custom data path
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import time

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(src_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pickle
import copy
from datetime import datetime
from agent_orchestrator import run_agentic_pipeline
from config import logger

# Cache directory for sharing results with Streamlit UI
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _make_picklable(obj, path="root"):
    """
    Recursively remove unpicklable objects (functions, lambdas, closures).
    Returns a cleaned copy safe for pickle.
    """
    if obj is None:
        return None
    
    # Skip functions/lambdas/methods entirely
    if callable(obj) and not isinstance(obj, type):
        return None
    
    # Skip any object that has a 'simulator' attribute (like ResponseCurveAnalyzer)
    if hasattr(obj, 'simulator'):
        return None
    
    # Handle dictionaries
    SKIP_KEYS = {"simulator", "all_simulators", "run_custom", "builder", "analyzer"}
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            cleaned = _make_picklable(v, f"{path}.{k}")
            if cleaned is not None or not callable(v):
                clean[k] = cleaned
        return clean
    
    # Handle lists
    if isinstance(obj, list):
        return [_make_picklable(item, f"{path}[{i}]") for i, item in enumerate(obj)]
    
    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(_make_picklable(item, f"{path}[{i}]") for i, item in enumerate(obj))
    
    # Primitive types and numpy/pandas objects are fine
    return obj


def save_results_to_cache(granularity, state):
    """
    Save pipeline state to cache so Streamlit can load it.
    
    Note: Functions (simulator, run_custom, etc.) cannot be pickled.
    We save everything else and rebuild functions on load.
    """
    try:
        cache_path = os.path.join(CACHE_DIR, f"mmix_results_{granularity}.pkl")
        
        # Create a shallow copy of state
        state_dict = {
            "granularity": state.granularity,
            "top_n_scenarios": state.top_n_scenarios,
            "model_filter": state.model_filter,
            "skip_narratives": getattr(state, "skip_narratives", False),
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "spec_strategy": state.spec_strategy,
            "reasoning_trace": state.reasoning_trace,
            "decisions": state.decisions,
            "quality_scores": state.quality_scores,
            "current_phase": state.current_phase,
            "paths": state.paths,
            # Data objects (DataFrames are picklable)
            "corr_matrix": state.corr_matrix,
            "outlier_log": state.outlier_log,
            "assumptions": state.assumptions,
        }
        
        # Clean model_result of functions
        if state.model_result:
            state_dict["model_result"] = _make_picklable(state.model_result)
        else:
            state_dict["model_result"] = None
            
        # Clean fe_result
        if state.fe_result:
            state_dict["fe_result"] = _make_picklable(state.fe_result)
        else:
            state_dict["fe_result"] = None
            
        # Clean aggregated_data
        if state.aggregated_data:
            state_dict["aggregated_data"] = _make_picklable(state.aggregated_data)
        else:
            state_dict["aggregated_data"] = None
            
        # Clean response_curves
        if state.response_curves:
            state_dict["response_curves"] = _make_picklable(state.response_curves)
        else:
            state_dict["response_curves"] = None
            
        # Narrator - just save the narratives dict if present
        if state.narrator and hasattr(state.narrator, "narratives"):
            state_dict["narrator_narratives"] = state.narrator.narratives
        else:
            state_dict["narrator_narratives"] = None
        
        with open(cache_path, "wb") as f:
            pickle.dump({
                "state_dict": state_dict,
                "timestamp": datetime.now().isoformat(),
                "granularity": granularity,
                "source": "main.py CLI",
                "version": 2,  # Version flag for loader compatibility
            }, f)
        logger.info(f"Results cached to: {cache_path}")
        print(f"  [CACHE] Results saved to: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Could not cache results: {e}")
        print(f"  [CACHE ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def run_both_granularities(top_n_scenarios=1, skip_eda=False,
                           skip_narratives=False, model_filter="all",
                           data_dir=None, output_dir=None):
    """
    Run agentic pipeline for both weekly and monthly, then compare.

    Both runs go through the full agentic loop independently.
    """
    start_time = time.time()
    results = {}

    for g in ['weekly', 'monthly']:
        print(f"\n{'='*70}")
        print(f"  RUNNING {g.upper()} GRANULARITY")
        print(f"{'='*70}")

        state = run_agentic_pipeline(
            granularity=g,
            top_n_scenarios=top_n_scenarios,
            model_filter=model_filter,
            skip_eda=skip_eda,
            skip_narratives=skip_narratives,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        results[g] = state
        
        # Cache results for Streamlit UI
        if state and state.model_result:
            save_results_to_cache(g, state)

    # Compare
    _compare_granularities(results)

    elapsed = time.time() - start_time
    print(f"\n  Total runtime (both): {elapsed:.1f} seconds")
    return results


def _compare_granularities(results):
    """Compare weekly vs monthly agentic results."""
    print("\n" + "=" * 70)
    print("[COMPARISON] WEEKLY vs MONTHLY")
    print("=" * 70)

    for g in ['weekly', 'monthly']:
        state = results.get(g)
        if state is None or state.model_result is None:
            print(f"\n  {g.upper()}: No results")
            continue

        best = state.model_result['best_model']
        print(f"\n  {g.upper()} (iterations={state.iteration}, strategy={state.spec_strategy}):")
        print(f"    Best: {best['spec_name']} | {best['model_type']} | {best['transform']}")
        print(f"    R2: {best['train_result']['r_squared']:.4f} | "
              f"Adj R2: {best['train_result']['adj_r_squared']:.4f}")
        cv_r2 = best.get('cv_result', {}).get('cv_r2')
        if cv_r2 is not None:
            print(f"    CV R2: {cv_r2:.4f}")
        print(f"    Composite: {best['scores']['composite']:.4f}")
        print(f"    Ordinality: {'PASS' if best['ordinality']['passed'] else 'FAIL'}")
        print(f"    Quality score: {state.quality_scores.get(state.iteration, {}).get('score', 'N/A')}")
        print(f"    Coefficients:")
        for feat, coef in best['train_result']['coefficients'].items():
            if feat == 'const':
                continue
            print(f"      {feat:35s} = {coef:+.4f}")

    # Insight agreement
    weekly_state = results.get('weekly')
    monthly_state = results.get('monthly')
    if (weekly_state and weekly_state.model_result and
        monthly_state and monthly_state.model_result):
        print("\n  Insight Agreement:")
        w_ins = weekly_state.model_result.get('convergence', {}).get('insights', {})
        m_ins = monthly_state.model_result.get('convergence', {}).get('insights', {})
        all_features = set(list(w_ins.keys()) + list(m_ins.keys()))
        for feat in sorted(all_features):
            w_dir = w_ins.get(feat, {}).get('direction', 'N/A')[:15]
            m_dir = m_ins.get(feat, {}).get('direction', 'N/A')[:15]
            agree = "AGREE" if w_dir == m_dir else "DIFFER"
            print(f"    {feat:35s} W: {w_dir:15s} M: {m_dir:15s} [{agree}]")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Agentic MMIX Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Weekly, agentic evaluation
  python main.py -g weekly -t 3               Weekly, top 3 scenarios
  python main.py -g both --skip-eda           Both granularities, skip EDA
  python main.py --data-dir ./my_data         Custom data path
        """
    )
    parser.add_argument(
        '--granularity', '-g',
        choices=['weekly', 'monthly', 'both'],
        default='weekly',
        help='Time granularity (default: weekly)'
    )
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip EDA for faster re-runs'
    )
    parser.add_argument(
        '--skip-narratives',
        action='store_true',
        help='Skip GenAI narrative generation'
    )
    parser.add_argument(
        '--top-models', '-t',
        type=int, default=1,
        help='Number of top models for scenarios (default: 1)'
    )
    parser.add_argument(
        '--data-dir',
        type=str, default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str, default=None,
        help='Path to output directory'
    )
    parser.add_argument(
        '--models', '-m',
        type=str, default='all',
        help='Model types: all, linear, or comma-separated (e.g. OLS,Ridge)'
    )
    args = parser.parse_args()

    if args.granularity == 'both':
        result = run_both_granularities(
            top_n_scenarios=args.top_models,
            skip_eda=args.skip_eda,
            skip_narratives=args.skip_narratives,
            model_filter=args.models,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    else:
        result = run_agentic_pipeline(
            granularity=args.granularity,
            top_n_scenarios=args.top_models,
            model_filter=args.models,
            skip_eda=args.skip_eda,
            skip_narratives=args.skip_narratives,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        
        # Cache results for Streamlit UI
        if result and result.model_result:
            save_results_to_cache(args.granularity, result)

    if result is None:
        print("\n[ERROR] Pipeline failed.")
        sys.exit(1)

    print("\n[DONE] Pipeline complete.")
    print(f"[INFO] Results cached to outputs/cache/ — run 'streamlit run app.py' and click 'Load Cached Results'")


if __name__ == "__main__":
    main()