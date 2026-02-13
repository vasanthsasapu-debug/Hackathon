"""
=============================================================================
LLM CONFIG — Azure OpenAI (GPT-4.1)
=============================================================================
Place this in: src/llm_config.py
Create a .env file in the Hackathon/ root with your API key.
=============================================================================
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://zs-eu1-ail-agentics-openai-team10.openai.azure.com/"
API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

def get_llm_client():
    """Initialize and return Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ AZURE_OPENAI_API_KEY not found in environment.\n"
            "   Create a .env file in the project root with:\n"
            "   AZURE_OPENAI_API_KEY=your_key_here"
        )

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        api_version=API_VERSION
    )
    print("✅ Azure OpenAI client initialized.")
    return client


def generate_narrative(client, prompt, system_prompt=None, max_tokens=1000, temperature=0.3):
    """
    Generate a business narrative using GPT-4.1.
    
    Args:
        client: AzureOpenAI client
        prompt: The user prompt (data/findings to explain)
        system_prompt: Optional system context
        max_tokens: Max response length
        temperature: Creativity (lower = more focused)
    
    Returns:
        str: Generated narrative text
    """
    if system_prompt is None:
        system_prompt = (
            "You are a senior marketing analytics consultant specializing in "
            "Marketing Mix Modeling (MMIX) for e-commerce. You explain data findings "
            "in clear, actionable business language. Be concise but insightful. "
            "Always highlight: what happened, why it matters, and what to do about it."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return f"[Narrative generation failed: {e}]"


def test_connection():
    """Quick test to verify Azure OpenAI connection works."""
    try:
        client = get_llm_client()
        response = generate_narrative(
            client,
            "Say 'Connection successful!' in one sentence."
        )
        print(f"🧪 Test response: {response}")
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
