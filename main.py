"""
=============================================================================
MMIX PIPELINE -- Main Entry Point
=============================================================================

Orchestrates end-to-end MMIX workflow with dynamic modes:
  - deterministic: Run all steps, no LLM (fast, cost-effective)
  - agentic: Run all steps + LLM narratives (rich output, higher cost)

Usage:
  python main.py --data data/Secondfile.csv --mode deterministic --output-dir outputs/
  python main.py --data data/Secondfile.csv --mode agentic --output-dir outputs/
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from typing import Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.orchestrator import PipelineState, run_pipeline
from agents.llm import get_llm_client


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MMIX Marketing Mix Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast deterministic run
  python main.py --data data/Secondfile.csv --mode deterministic
  
  # Full agentic run with narratives
  python main.py --data data/Secondfile.csv --mode agentic --output-dir outputs/
  
  # Custom output directory
  python main.py --data data/Sales.csv --mode deterministic --output-dir results/
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/Secondfile.csv",
        help="Path to input CSV file (default: data/Secondfile.csv)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "agentic"],
        default="deterministic",
        help="Execution mode: deterministic (fast, no LLM) or agentic (with narratives)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: Path to checkpoint JSON to resume from"
    )
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file with error handling.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Pandas DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or corrupted
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    print(f"📂 Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {str(e)}")


# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

def setup_output_directory(output_dir: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path
        
    Returns:
        Path object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    return output_path


# =============================================================================
# MODE VALIDATION
# =============================================================================

def validate_mode(mode: str) -> None:
    """
    Validate execution mode and check prerequisites.
    
    Args:
        mode: "deterministic" or "agentic"
        
    Raises:
        ValueError: If agentic mode but Azure OpenAI not configured
    """
    if mode == "agentic":
        print("🤖 Agentic mode: Will generate LLM narratives")
        try:
            client = get_llm_client()
            print("   ✅ Azure OpenAI client initialized")
        except ValueError as e:
            print(f"   ❌ Azure OpenAI not configured: {str(e)}")
            print("   💡 Falling back to deterministic mode")
    elif mode == "deterministic":
        print("⚡ Deterministic mode: Fast execution, no LLM")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 80)
    print(" MMIX MARKETING MIX MODELING PIPELINE")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\n📋 Configuration:")
    print(f"   Data file: {args.data}")
    print(f"   Mode: {args.mode}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Verbose: {args.verbose}")
    
    # Load data
    print(f"\n{'─' * 80}")
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        sys.exit(1)
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    # Validate mode
    print(f"\n{'─' * 80}")
    validate_mode(args.mode)
    
    # Initialize pipeline state
    print(f"\n{'─' * 80}")
    print("🔧 Initializing pipeline state...")
    
    state = PipelineState(
        data={"main": df},
        mode=args.mode,
        output_dir=str(output_path),
        verbose=args.verbose
    )
    state.log("Pipeline initialized")
    
    # Run pipeline
    print(f"\n{'─' * 80}")
    print("🚀 Running pipeline...\n")
    
    # Initialize LLM client if agentic mode
    llm_client = None
    if args.mode == "agentic":
        try:
            llm_client = get_llm_client()
        except ValueError as e:
            print(f"   ⚠️  {str(e)}")
            print("   💡 Falling back to deterministic mode")
            state.mode = "deterministic"
    
    try:
        state = run_pipeline(state, llm_client)
        print(f"\n{'─' * 80}")
        print("✅ Pipeline completed successfully!")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Steps completed: {len(state.completed_steps)}")
        for step in state.completed_steps:
            print(f"     ✓ {step}")
        
        if state.errors:
            print(f"\n   Warnings/Errors ({len(state.errors)}):")
            for error in state.errors:
                print(f"     ⚠️  {error}")
        
        print(f"\n   Output directory: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        
        if state.errors:
            print(f"\n   Errors encountered:")
            for error in state.errors:
                print(f"     • {error}")
        
        sys.exit(1)
    
    print("\n" + "=" * 80)


# =============================================================================
# LEGACY SUPPORT (for backward compatibility)
# =============================================================================

def run_full_mmix_pipeline(
    data_dir: str = "data",
    enable_llm: bool = True,
    export_excel: bool = True,
    export_ppt: bool = True,
    output_dir: str = "outputs"
) -> Dict:
    """
    LEGACY: Run the complete MMIX pipeline (old interface).
    
    TODO (V2): Remove after migrating to new interface.
    """
    print("\n⚠️  Using legacy run_full_mmix_pipeline() interface")
    print("   Recommend: Use new main.py with args instead\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Step 1: Loading Data")
    print("-" * 80)
    data_dir_path = Path(__file__).parent / data_dir
    
    try:
        df = pd.read_csv(data_dir_path / "Secondfile.csv")
        print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return {"status": "failed", "error": str(e)}
    
    # Initialize LLM client (optional)
    llm_client = None
    if enable_llm:
        print("Step 2: Initializing LLM")
        print("-" * 80)
        try:
            llm_client = get_llm_client()
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {str(e)}")
    
    # Run new pipeline
    print("Step 3: Running Agentic Pipeline")
    print("-" * 80)
    
    try:
        state = PipelineState(
            data={"main": df},
            mode="agentic" if enable_llm else "deterministic",
            output_dir=output_dir,
        )
        state = run_pipeline(state)
        print("✅ Pipeline completed")
        
        return {
            "status": "success",
            "state": state,
        }
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

