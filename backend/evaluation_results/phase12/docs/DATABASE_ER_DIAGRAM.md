# HalluciSense Database Entity-Relationship Diagram

```
Organizations 1 ──── N Users 1 ──── N APIKeys
     1                    │
     │                    │
     N                    N
  Projects 1 ──── N VerificationSessions 1 ──── N ProviderResponses
```
