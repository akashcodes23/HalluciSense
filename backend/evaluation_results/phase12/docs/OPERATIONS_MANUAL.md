# HalluciSense Operations & Maintenance Manual

## System Health Check
`GET /api/v1/pillar2/health`

## Prometheus Metrics
`GET /metrics`

## Restarting Services
```bash
docker-compose restart api celery_worker
```
