#!/usr/bin/env bash
# HalluciSense cURL API Verification Example

curl -X POST "http://localhost:8000/api/v1/verification/verify-text" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Paris is the capital and most populous city of France."
     }'
