# HalluciSense v1 REST API Reference

Base Endpoint: `https://api.hallucisense.ai/api/v1`

## POST `/pillar2/verify`
Verify response text for factual grounding.

### Request Headers
- `Content-Type`: `application/json`
- `X-API-Key`: `hs_live_...`

### Request Body
```json
{
  "text": "Quantum computing uses qubits to calculate states.",
  "pillar1_probability": 0.15
}
```

### Response
```json
{
  "verification_id": "verif_101",
  "hallucisense_score": {
    "hallucisense_score": 12.50,
    "risk_category": "VERY_LOW",
    "overall_confidence": 0.972
  }
}
```
