#!/usr/bin/env python3
"""Check if narratives are being generated in agentic mode."""

import sys
sys.path.insert(0, "/Users/kunalbhargava/GitHub/Hackathon")

from agents.orchestrator import PipelineState, run_pipeline
from agents.llm import get_llm_client
import pandas as pd

df = pd.read_csv("/Users/kunalbhargava/GitHub/Hackathon/data/Secondfile.csv")
state = PipelineState(mode="agentic", verbose=False, data={"main": df})

try:
    llm_client = get_llm_client()
    print("✅ LLM client obtained")
except Exception as e:
    print(f"⚠️  LLM client failed: {e}")
    llm_client = None

state = run_pipeline(state, llm_client=llm_client)

print("\n" + "="*80)
print("✅ Checking generated narratives:")
print("="*80)
print(f"eda_narrative: {repr(state.eda_narrative)[:100]}")
print(f"outlier_narrative: {repr(state.outlier_narrative)[:100]}")
print(f"feature_narrative: {repr(state.feature_narrative)[:100]}")
print(f"model_narrative: {repr(state.model_narrative)[:100]}")
print(f"optimization_narrative: {repr(state.optimization_narrative)[:100]}")
