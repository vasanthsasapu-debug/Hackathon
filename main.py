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

from agent_orchestrator import run_agentic_pipeline
from config import logger


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
            data_dir=data_dir,
            output_dir=output_dir,
        )
        results[g] = state

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
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )

    if result is None:
        print("\n[ERROR] Pipeline failed.")
        sys.exit(1)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
