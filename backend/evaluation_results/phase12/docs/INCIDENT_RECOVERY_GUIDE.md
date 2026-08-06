# HalluciSense Disaster & Incident Recovery Guide

1. **Database Restore**: Restore latest Neon PostgreSQL Point-in-Time snapshot.
2. **Cache Purge**: `redis-cli FLUSHALL` to reset expired evidence caches.
3. **Pillar 1 Firewall Check**: Verify `sha256sum evaluation_results/phase6k/final_model/pillar1_logistic_model.joblib`.
