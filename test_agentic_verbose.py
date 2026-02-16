#!/usr/bin/env python3
"""Test agentic mode with explicit verbose logging of LLM calls."""

import sys
sys.path.insert(0, "/Users/kunalbhargava/GitHub/Hackathon")

from agents.orchestrator import PipelineState, run_pipeline
from agents.llm import get_llm_client
import pandas as pd

# Load data
df = pd.read_csv("/Users/kunalbhargava/GitHub/Hackathon/data/Secondfile.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Initialize state with verbose=True to see log messages
state = PipelineState(mode="agentic", verbose=True, data={"main": df})

# Get LLM client
try:
    llm_client = get_llm_client()
    print("✅ LLM client obtained")
except Exception as e:
    print(f"❌ Failed to get LLM client: {e}")
    llm_client = None

# Run pipeline with agentic mode
print("\n" + "="*80)
print("Running pipeline with agentic mode...")
print("="*80 + "\n")

state = run_pipeline(state, llm_client=llm_client)

print("\n" + "="*80)
print("Pipeline completed")
print("="*80)
print(f"\nLogs captured ({len(state.step_logs)} lines):")
for log in state.step_logs:
    print(f"  {log}")

if state.errors:
    print(f"\nErrors ({len(state.errors)} lines):")
    for err in state.errors:
        print(f"  {err}")
