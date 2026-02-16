"""
=============================================================================
MAIN ENTRY POINT -- Complete Agentic MMIX Pipeline
=============================================================================
Orchestrates all 3 components:
  1. Core Pipeline (EDA, Outlier Detection, Features, Modeling)
  2. Agentic Wrapper (LLM Integration, State Management, Feedback Loops)
  3. Polish (Optimization, Export, Reporting)
=============================================================================
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from typing import Dict, Optional

# Import components
from mmix.agents.orchestrator import Orchestrator
from mmix.agents.llm import get_llm_client
from mmix.export.excel import export_to_excel
from mmix.export.powerpoint import generate_ppt_presentation


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_directory(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from data directory.
    
    Args:
        data_dir: Path to data folder
        
    Returns:
        {filename: dataframe}
    """
    data = {}
    
    file_mapping = {
        "firstfile.csv": "transactions",
        "Sales.csv": "sales",
        "Secondfile.csv": "monthly",
        "SpecialSale.csv": "special_sales",
        "MediaInvestment.csv": "media_investment",
        "MonthlyNPSscore.csv": "nps",
        "ProductList.csv": "products",
    }
    
    for filename, label in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                if filename == "Sales.csv":
                    # Tab-delimited
                    df = pd.read_csv(filepath, sep="\t", parse_dates=["Date"])
                else:
                    df = pd.read_csv(filepath)
                
                data[label] = df
                print(f"✅ Loaded {label}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
        else:
            print(f"⚠️  {filename} not found")
    
    return data


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_mmix_pipeline(
    data_dir: str = "data",
    enable_llm: bool = True,
    export_excel: bool = True,
    export_ppt: bool = True,
    output_dir: str = "outputs"
) -> Dict:
    """
    Run the complete Agentic MMIX pipeline.
    
    Args:
        data_dir: Path to input data folder
        enable_llm: Include GenAI narratives
        export_excel: Export results to Excel
        export_ppt: Export results to PowerPoint
        output_dir: Directory for outputs
        
    Returns:
        {
            "state": PipelineState,
            "data": loaded data,
            "status": success/failure
        }
    """
    print("\n" + "="*80)
    print("AGENTIC MMIX PIPELINE - STARTING")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Step 1: Loading Data")
    print("-" * 80)
    data_dir_path = os.path.join(os.path.dirname(__file__), data_dir)
    data = load_data_from_directory(data_dir_path)
    
    if not data:
        print("❌ No data loaded. Exiting.")
        return {"status": "failed", "error": "No data loaded"}
    
    print()
    
    # Initialize LLM client (optional)
    llm_client = None
    if enable_llm:
        print("Step 2: Initializing LLM")
        print("-" * 80)
        try:
            llm_client = get_llm_client()
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {str(e)}")
            print("   Continuing without LLM narratives...")
        print()
    
    # Run orchestrator
    print("Step 3: Running Agentic Pipeline")
    print("-" * 80)
    orchestrator = Orchestrator(data, llm_client)
    state = orchestrator.run_full_pipeline(auto_feedback_loop=True)
    
    print()
    
    # Summary
    print("Step 4: Pipeline Summary")
    print("-" * 80)
    orchestrator.print_summary()
    
    print()
    
    # Export state
    state_json_path = os.path.join(output_dir, "pipeline_state.json")
    orchestrator.export_state_to_json(state_json_path)
    
    # Export Excel (if requested)
    if export_excel:
        print("Step 5: Exporting to Excel")
        print("-" * 80)
        excel_path = os.path.join(output_dir, "MMIX_Analysis_Results.xlsx")
        
        export_to_excel(
            excel_path,
            eda_results=state.eda_results if state.eda_results else None,
            ranked_models=state.ranked_models if state.ranked_models else None,
            elasticities=state.elasticities if state.elasticities else None,
            scenarios=state.optimization_scenarios if state.optimization_scenarios else None,
            narratives={
                "EDA": state.eda_narrative,
                "Outlier Removal": state.outlier_narrative,
                "Feature Engineering": state.feature_narrative,
                "Models": state.model_narrative,
                "Optimization": state.optimization_narrative,
            }
        )
        print()
    
    # Export PowerPoint (if requested)
    if export_ppt:
        print("Step 6: Exporting to PowerPoint")
        print("-" * 80)
        ppt_path = os.path.join(output_dir, "MMIX_Analysis_Report.pptx")
        
        eda_narratives = {}
        if state.eda_results:
            for segment_name in state.eda_results.keys():
                eda_narratives[segment_name] = state.eda_narrative
        
        generate_ppt_presentation(
            ppt_path,
            eda_narratives=eda_narratives if eda_narratives else {"National": state.eda_narrative},
            outlier_narrative=state.outlier_narrative,
            feature_narrative=state.feature_narrative,
            model_narrative=state.model_narrative,
            optimization_narrative=state.optimization_narrative,
            recommendations=[
                "Prioritize high-elasticity channels for budget allocation",
                "Implement quarterly model rebalancing based on response curves",
                "Monitor ordinality constraints to ensure realistic predictions",
                "Consider blue-sky scenario for strategic long-term planning",
            ]
        )
        print()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE ✅")
    print("="*80)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}")
    print(f"  - pipeline_state.json")
    if export_excel:
        print(f"  - MMIX_Analysis_Results.xlsx")
    if export_ppt:
        print(f"  - MMIX_Analysis_Report.pptx")
    print()
    
    return {
        "status": "success",
        "state": state,
        "data": data,
        "output_dir": output_dir,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic MMIX Pipeline")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="outputs", help="Path to output directory")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM narratives")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    parser.add_argument("--no-ppt", action="store_true", help="Skip PowerPoint export")
    
    args = parser.parse_args()
    
    result = run_full_mmix_pipeline(
        data_dir=args.data_dir,
        enable_llm=not args.no_llm,
        export_excel=not args.no_excel,
        export_ppt=not args.no_ppt,
        output_dir=args.output_dir,
    )
    
    sys.exit(0 if result["status"] == "success" else 1)
