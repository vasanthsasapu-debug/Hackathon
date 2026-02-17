"""
=============================================================================
MAIN RUNNER -- Agentic E-Commerce MMIX Pipeline
=============================================================================
Usage:
    python main.py                                    # Default: weekly, top 1
    python main.py -g weekly -t 3                     # Weekly, top 3 models
    python main.py -g monthly                         # Monthly
    python main.py -g both -t 3 --skip-eda            # Both, skip EDA
    python main.py --data-dir /path/to/data           # Custom data path
    python main.py --output-dir /path/to/outputs      # Custom output path
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import time
import traceback

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(src_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import get_paths, PipelineSummary
from eda_pipeline import load_all_data, run_full_eda
from outlier_detection import run_outlier_pipeline
from data_aggregation import build_modeling_dataset
from feature_engineering import run_feature_engineering
from modeling_engine import run_modeling_pipeline


def run_pipeline(granularity='weekly', skip_eda=False, top_n_scenarios=1,
                 data_dir=None, output_dir=None):
    """
    Run the complete MMIX pipeline.

    Args:
        granularity: 'weekly', 'monthly', or 'both'
        skip_eda: skip full EDA for faster re-runs
        top_n_scenarios: how many top models run scenarios
        data_dir: path to data folder (auto-detects if None)
        output_dir: path to output folder (auto-detects if None)
    """
    start_time = time.time()
    paths = get_paths(data_dir, output_dir)
    summary_collector = PipelineSummary()

    # Ensure output dirs exist
    os.makedirs(paths['plots_dir'], exist_ok=True)
    os.makedirs(paths['reports_dir'], exist_ok=True)

    print("=" * 70)
    print(f"  AGENTIC MMIX PIPELINE")
    print(f"  Granularity: {granularity.upper()} | Top models for scenarios: {top_n_scenarios}")
    print(f"  Data: {paths['data_dir']}")
    print(f"  Output: {paths['output_dir']}")
    print("=" * 70)

    # =================================================================
    # PHASE 1: DATA LOADING
    # =================================================================
    print("\n" + "=" * 70)
    print("[PHASE 1] DATA LOADING")
    print("=" * 70)
    try:
        data = load_all_data(paths['data_dir'])
        if not data:
            print("[ERROR] No data loaded. Check data directory.")
            return None
        summary_collector.add_step("data_loading",
            f"Loaded {len(data)} datasets from {paths['data_dir']}")
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        traceback.print_exc()
        return None

    # =================================================================
    # PHASE 2: EDA (optional)
    # =================================================================
    if not skip_eda:
        print("\n" + "=" * 70)
        print("[PHASE 2] EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        try:
            eda_data, classifications, issues, corr_matrix = run_full_eda(
                data_dir=paths['data_dir'], save_dir=paths['plots_dir']
            )
            summary_collector.add_step("eda",
                f"EDA complete. {len(issues)} issues found. "
                f"Correlation matrix computed for {len(corr_matrix) if corr_matrix is not None else 0} variables.")
        except Exception as e:
            print(f"[ERROR] EDA failed: {e}")
            traceback.print_exc()
            summary_collector.add_error("eda", str(e))
            print("[INFO] Continuing without EDA...")
    else:
        print("\n[PHASE 2] EDA skipped (--skip-eda)")

    # =================================================================
    # PHASE 3: OUTLIER DETECTION
    # =================================================================
    print("\n" + "=" * 70)
    print("[PHASE 3] OUTLIER DETECTION & CLEANING")
    print("=" * 70)
    try:
        clean_data, outlier_log, assumptions = run_outlier_pipeline(
            data, granularity=granularity, save_dir=paths['plots_dir']
        )
        summary_collector.add_step("outlier_detection",
            f"Cleaned {len(clean_data)} datasets. {len(outlier_log)} actions taken. "
            f"{len(assumptions)} assumptions documented.",
            log=[entry.get('reason', '') for entry in outlier_log])
    except Exception as e:
        print(f"[ERROR] Outlier detection failed: {e}")
        traceback.print_exc()
        return None

    # =================================================================
    # PHASE 4+: RUN PER GRANULARITY
    # =================================================================
    print("\n" + "=" * 70)
    print(f"[PHASE 4] DATA AGGREGATION ({granularity.upper()})")
    print("=" * 70)

    if granularity == 'both':
        results = {}
        for g in ['weekly', 'monthly']:
            print(f"\n{'='*50}")
            print(f"  Running {g.upper()}")
            print(f"{'='*50}")
            result = _run_single_granularity(
                clean_data, g, top_n_scenarios, paths, summary_collector
            )
            results[g] = result
        _compare_granularities(results)
        elapsed = time.time() - start_time
        print(f"\n  Total runtime: {elapsed:.1f} seconds")
        results['summary_collector'] = summary_collector
        return results
    else:
        result = _run_single_granularity(
            clean_data, granularity, top_n_scenarios, paths, summary_collector
        )
        elapsed = time.time() - start_time
        print(f"\n  Total runtime: {elapsed:.1f} seconds")
        if result:
            result['summary_collector'] = summary_collector
        return result


def _run_single_granularity(clean_data, granularity, top_n_scenarios, paths, summary_collector):
    """Run aggregation, feature engineering, and modeling for one granularity."""

    # --- Aggregation ---
    try:
        agg_result = build_modeling_dataset(clean_data, granularity=granularity)
        if agg_result is None:
            print(f"[ERROR] Aggregation failed for {granularity}")
            summary_collector.add_error(f"aggregation_{granularity}", "Aggregation returned None")
            return None
        aggregated_data = agg_result['data']
        n_periods = agg_result['n_periods']
        summary_collector.add_step(f"aggregation_{granularity}", agg_result.get('summary', ''))
        print(f"  {agg_result.get('summary', '')}")
    except Exception as e:
        print(f"[ERROR] Aggregation failed: {e}")
        traceback.print_exc()
        summary_collector.add_error(f"aggregation_{granularity}", str(e))
        return None

    # --- Feature Engineering ---
    print("\n" + "=" * 70)
    print(f"[PHASE 5] FEATURE ENGINEERING ({granularity.upper()}, n={n_periods})")
    print("=" * 70)
    try:
        fe_result = run_feature_engineering(aggregated_data, save_dir=paths['plots_dir'])
        if fe_result is None:
            print(f"[ERROR] Feature engineering failed")
            summary_collector.add_error(f"feature_engineering_{granularity}", "Returned None")
            return None
        # Collect summaries from all FE steps
        for step_name, step_summary in fe_result.get('summaries', {}).items():
            summary_collector.add_step(f"fe_{granularity}_{step_name}", step_summary)
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        traceback.print_exc()
        summary_collector.add_error(f"feature_engineering_{granularity}", str(e))
        return None

    # --- Modeling ---
    print("\n" + "=" * 70)
    print(f"[PHASE 6] MODELING ({granularity.upper()}, n={n_periods})")
    print("=" * 70)
    try:
        model_result = run_modeling_pipeline(
            fe_result, clean_data=aggregated_data,
            save_dir=paths['plots_dir'], top_n_scenarios=top_n_scenarios
        )
        if model_result is None:
            print(f"[ERROR] Modeling failed")
            summary_collector.add_error(f"modeling_{granularity}", "Returned None")
            return None
        # Collect modeling summaries
        for step_name, step_summary in model_result.get('summaries', {}).items():
            summary_collector.add_step(f"model_{granularity}_{step_name}", step_summary)
    except Exception as e:
        print(f"[ERROR] Modeling failed: {e}")
        traceback.print_exc()
        summary_collector.add_error(f"modeling_{granularity}", str(e))
        return None

    return {
        "granularity": granularity,
        "n_periods": n_periods,
        "aggregation": agg_result,
        "feature_engineering": fe_result,
        "modeling": model_result
    }


def _compare_granularities(results):
    """Compare weekly vs monthly results."""
    print("\n" + "=" * 70)
    print("[COMPARISON] WEEKLY vs MONTHLY")
    print("=" * 70)

    for g in ['weekly', 'monthly']:
        r = results.get(g)
        if r is None or r.get('modeling') is None:
            print(f"\n  {g.upper()}: No results")
            continue
        best = r['modeling']['best_model']
        print(f"\n  {g.upper()} (n={r['n_periods']}):")
        print(f"    Best: {best['spec_name']} | {best['model_type']} | {best['transform']}")
        print(f"    R2: {best['train_result']['r_squared']:.4f} | "
              f"Adj R2: {best['train_result']['adj_r_squared']:.4f}")
        if best['cv_result'].get('cv_r2') is not None:
            print(f"    CV R2: {best['cv_result']['cv_r2']:.4f}")
        print(f"    Composite: {best['scores']['composite']:.4f}")
        print(f"    Ordinality: {'PASS' if best['ordinality']['passed'] else 'FAIL'}")
        print(f"    Coefficients:")
        for feat, coef in best['train_result']['coefficients'].items():
            if feat == 'const':
                continue
            print(f"      {feat:35s} = {coef:+.4f}")

    # Insight agreement
    print("\n  Insight Agreement:")
    weekly_conv = results.get('weekly', {})
    monthly_conv = results.get('monthly', {})
    if weekly_conv and monthly_conv:
        w_ins = weekly_conv.get('modeling', {}).get('convergence', {}).get('insights', {})
        m_ins = monthly_conv.get('modeling', {}).get('convergence', {}).get('insights', {})
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
  python main.py                              Weekly, 1 model scenarios
  python main.py -g weekly -t 3               Weekly, top 3 models
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
    args = parser.parse_args()

    result = run_pipeline(
        granularity=args.granularity,
        skip_eda=args.skip_eda,
        top_n_scenarios=args.top_models,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    if result is None:
        print("\n[ERROR] Pipeline failed.")
        sys.exit(1)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
