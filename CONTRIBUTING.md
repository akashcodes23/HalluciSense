# Contributing to HalluciSense

Thank you for your interest in contributing to HalluciSense! We welcome contributions from researchers, software engineers, and open science advocates.

---

## Development Workflow

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/akashcodes23/HalluciSense.git
   cd HalluciSense
   ```

2. **Environment Setup**:
   ```bash
   bash scripts/fresh_install.sh
   ```

3. **Running Pytest Tests**:
   ```bash
   cd backend
   pytest tests/ -v
   ```

4. **Submitting Pull Requests**:
   Ensure all CI Quality Gates pass (`python3 backend/scripts/check_phase26_quality_gates.py`) before opening a PR.
