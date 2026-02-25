"""
=============================================================================
demo_scenarios.py -- Interactive Scenario Demo
=============================================================================
Run the pipeline and simulate custom what-if scenarios from the terminal.

Usage:
    python demo_scenarios.py                                  # Run pipeline + interactive
    python demo_scenarios.py -g weekly                        # Weekly granularity
    python demo_scenarios.py Online.marketing=50 TV=-10       # Custom scenario
    python demo_scenarios.py Sponsorship=30 --sale            # With sale event
    python demo_scenarios.py --skip-eda Online.marketing=100  # Skip EDA + scenario
    python demo_scenarios.py --preset diwali                  # Named preset
    python demo_scenarios.py -q Online.marketing=50           # Quiet — only show results
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import io
import logging

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agent_orchestrator import run_agentic_pipeline
from modeling_engine import run_custom_scenario

# Named presets for quick demo
PRESETS = {
    "diwali": {
        "desc": "Diwali strategy: +100% digital + sale event",
        "channels": {"Online.marketing": 100, "Affiliates": 80, "SEM": 50, "Sponsorship": 30},
        "sale": 1,
    },
    "digital_push": {
        "desc": "Double digital performance spend",
        "channels": {"Online.marketing": 100, "Affiliates": 100, "SEM": 50},
        "sale": None,
    },
    "budget_cut": {
        "desc": "Cut all budgets 30%",
        "channels": {"TV": -30, "Digital": -30, "Sponsorship": -30, "Content.Marketing": -30,
                     "Online.marketing": -30, "Affiliates": -30, "SEM": -30, "Radio": -30, "Other": -30},
        "sale": None,
    },
    "tv_to_digital": {
        "desc": "Shift 20% from TV to Online.marketing",
        "channels": {"TV": -20, "Online.marketing": 20},
        "sale": None,
    },
    "sale_only": {
        "desc": "Activate sale event with no spend change",
        "channels": {},
        "sale": 1,
    },
}


def parse_channel_args(args):
    """Parse channel=value arguments from CLI."""
    channels = {}
    for arg in args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            try:
                channels[key.strip()] = float(val.strip())
            except ValueError:
                print(f"  [WARN] Ignoring invalid argument: {arg}")
    return channels


def run_pipeline_quiet(granularity, skip_eda):
    """Run the pipeline with all output suppressed."""
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    logging.getLogger("mmix").setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    try:
        state = run_agentic_pipeline(
            granularity=granularity,
            skip_eda=skip_eda,
        )
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        logging.getLogger("mmix").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.WARNING)

    return state


def print_model_summary(state):
    """Print just the best model info."""
    best = state.model_result["best_model"]
    tr = best["train_result"]
    cv = best.get("cv_result", {})

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL")
    print(f"{'=' * 60}")
    print(f"  Spec:       {best['spec_name']}")
    print(f"  Type:       {best['model_type']} ({best['transform']})")
    print(f"  R\u00B2:         {tr['r_squared']:.4f}")
    print(f"  Adj R\u00B2:     {tr['adj_r_squared']:.4f}")
    if cv.get("cv_r2") is not None:
        print(f"  CV R\u00B2:      {cv['cv_r2']:.4f}")
        print(f"  CV MAPE:    {cv['cv_mape']:.2f}%")
    print(f"  Ordinality: {'PASS' if best['ordinality']['passed'] else 'FAIL'}")
    print(f"  Iterations: {state.iteration}")
    print(f"  Strategy:   {state.spec_strategy}")
    print(f"\n  Coefficients:")
    for f, c in tr["coefficients"].items():
        if f == "const":
            continue
        print(f"    {f:35s} {c:+.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MMIX Scenario Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_scenarios.py Online.marketing=50 TV=-10       +50% online, -10% TV
  python demo_scenarios.py Sponsorship=30 --sale            +30% sponsorship + sale
  python demo_scenarios.py --preset diwali                  Named preset
  python demo_scenarios.py -q Online.marketing=50           Quiet mode
  python demo_scenarios.py --list-presets                   Show all presets

Available channels:
  TV, Digital, Sponsorship, Content.Marketing, Online.marketing,
  Affiliates, SEM, Radio, Other
        """
    )
    parser.add_argument("-g", "--granularity", choices=["weekly", "monthly"],
                        default="monthly", help="Time granularity (default: monthly)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA for faster run")
    parser.add_argument("--sale", action="store_true", help="Activate sale event in scenario")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress pipeline output — only show best model and scenario results")
    parser.add_argument("--preset", type=str, default=None,
                        help=f"Named preset: {', '.join(PRESETS.keys())}")
    parser.add_argument("--list-presets", action="store_true", help="Show available presets")
    parser.add_argument("channels", nargs="*", help="Channel changes as Channel=Pct (e.g. TV=-10)")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\n  Available presets:")
        for name, p in PRESETS.items():
            channels_str = ", ".join(f"{k}={v:+.0f}%" for k, v in p["channels"].items())
            sale_str = " + sale" if p["sale"] else ""
            print(f"    --preset {name:15s}  {p['desc']}")
            print(f"      {channels_str}{sale_str}")
        return

    # --- Run the pipeline ---
    if args.quiet:
        print("  Running pipeline (quiet)...", end="", flush=True)
        state = run_pipeline_quiet(args.granularity, args.skip_eda)
        print(" done.")
    else:
        print("\n" + "=" * 70)
        print("  RUNNING PIPELINE...")
        print("=" * 70)
        state = run_agentic_pipeline(
            granularity=args.granularity,
            skip_eda=args.skip_eda,
        )

    if state is None or state.model_result is None:
        print("\n[ERROR] Pipeline failed.")
        sys.exit(1)

    sim = state.model_result.get("simulator")
    if sim is None:
        print("\n[ERROR] No simulator available.")
        sys.exit(1)

    # --- Show best model ---
    print_model_summary(state)

    # --- Run preset or CLI scenario ---
    if args.preset:
        if args.preset not in PRESETS:
            print(f"\n[ERROR] Unknown preset '{args.preset}'. Use --list-presets.")
            sys.exit(1)
        preset = PRESETS[args.preset]
        print(f"  Preset: {preset['desc']}")
        sale = preset["sale"] if preset["sale"] is not None else (1 if args.sale else None)
        run_custom_scenario(sim, preset["channels"], sale_flag=sale)

    elif args.channels:
        channels = parse_channel_args(args.channels)
        if channels or args.sale:
            sale = 1 if args.sale else None
            run_custom_scenario(sim, channels, sale_flag=sale)
        else:
            print("  No valid channel arguments found.")

    else:
        # No scenario specified — run all presets
        for name, preset in PRESETS.items():
            print(f"  --- {preset['desc']} ---")
            run_custom_scenario(sim, preset["channels"], sale_flag=preset["sale"])
            print()

    # --- Interactive mode (skip in quiet+scenario mode) ---
    if not (args.quiet and (args.preset or args.channels)):
        print("\n" + "=" * 60)
        print("  INTERACTIVE MODE")
        print("=" * 60)
        print("  Examples:")
        print("    run_custom_scenario(sim, {'Online.marketing': 50})")
        print("    run_custom_scenario(sim, {'TV': -10}, sale_flag=1)")
        print("  Type exit() to quit.\n")

        import code
        code.interact(
            banner="",
            local={
                "sim": sim,
                "run_custom_scenario": run_custom_scenario,
                "state": state,
            }
        )


if __name__ == "__main__":
    main()