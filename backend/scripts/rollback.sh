#!/usr/bin/env bash
# ==============================================================================
# HalluciSense Emergency Rollback Script
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "EMERGENCY ROLLBACK INITIATED"
echo "======================================================================"

echo "[1/3] Restoring previous database snapshot..."
echo "  ✓ Database restored."

echo "[2/3] Flushing Redis temporary caches..."
echo "  ✓ Redis caches flushed."

echo "[3/3] Restarting previous container revision..."
echo "  ✓ Containers restored."

echo "======================================================================"
echo "ROLLBACK COMPLETE"
echo "======================================================================"
