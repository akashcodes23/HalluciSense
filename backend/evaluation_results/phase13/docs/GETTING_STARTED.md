# Getting Started with HalluciSense

Welcome to HalluciSense v1.0!

## Quick Install (Python SDK)
```bash
pip install hallucisense-sdk
```

## Quick Verification Code
```python
from hallucisense_sdk import HalluciSenseClient

client = HalluciSenseClient(api_key="hs_live_your_key_here")
result = client.verify("Albert Einstein discovered relativity in 1905.")

print(f"H-Score: {result.hallucisense_score:.2f}")
print(f"Risk Level: {result.risk_category}")
```
