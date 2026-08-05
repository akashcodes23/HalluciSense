"""
Independent Gemini API diagnostic script for HalluciSense.
Tests:
1. Config & API key loading
2. Model listing via google-generativeai
3. Non-streaming content generation
4. Streaming content generation
"""
import os
os.environ["GRPC_DNS_RESOLVER"] = "native"

import google.generativeai as genai
from app.core.config import settings

def main():
    print("==========================================================")
    print("HALLUCISENSE GEMINI DIAGNOSTIC SCRIPT")
    print("==========================================================")

    api_key = settings.GEMINI_API_KEY
    print("1. API Key Loaded:", bool(api_key))
    if not api_key:
        print("ERROR: GEMINI_API_KEY is missing from environment/settings!")
        return

    genai.configure(api_key=api_key)

    print("\n2. Listing Available Models for generateContent:")
    try:
        models = [m.name.replace("models/", "") for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for m in models[:10]:
            print("   -", m)
    except Exception as e:
        print("ERROR listing models:", e)
        return

    print("\n3. Testing Non-Streaming Generation ('gemini-2.0-flash')...")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content("What is Machine Learning in one short sentence?")
        print("   Response:", res.text.strip())
    except Exception as e:
        print("ERROR during non-streaming generation:", e)

    print("\n4. Testing Streaming Generation ('gemini-2.0-flash')...")
    try:
        model_stream = genai.GenerativeModel("gemini-2.0-flash")
        res_stream = model_stream.generate_content("Count from 1 to 5.", stream=True)
        chunks = []
        for chunk in res_stream:
            text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
            chunks.append(text)
            print(f"   [Chunk]: '{text.strip()}'")
        print("   Full Streamed Response:", "".join(chunks).strip())
    except Exception as e:
        print("ERROR during streaming generation:", e)

    print("\n==========================================================")
    print("DIAGNOSTIC TEST COMPLETE!")
    print("==========================================================")

if __name__ == "__main__":
    main()