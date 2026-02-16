#!/usr/bin/env python3
"""Test Azure key loading and LLM calls."""

import os
from dotenv import load_dotenv

# Load from explicit path
load_dotenv("/Users/kunalbhargava/GitHub/Hackathon/.env")

key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

print(f"✅ Key loaded: {key[:20]}..." if key else "❌ Key NOT loaded")
print(f"✅ Endpoint loaded: {endpoint}" if endpoint else "❌ Endpoint NOT loaded")

# Now try to create client
try:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version="2024-12-01-preview"
    )
    print("✅ Azure OpenAI client created successfully")
    
    # Try to call the API
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10,
        temperature=0.3
    )
    print(f"✅ LLM call succeeded: {response.choices[0].message.content[:50]}")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)[:300]}")
