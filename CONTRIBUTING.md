# Contributing to HalluciSense

Thank you for your interest in contributing to HalluciSense! We welcome bug fixes, documentation improvements, and engineering optimizations.

---

## 1. Scientific Integrity Policy (Strict Invariant)

> [!IMPORTANT]
> The scientific architecture, fusion equations, empirical weights ($\alpha=0.45, \beta=0.30, \gamma=0.25$), calibration models, and benchmark evaluation datasets (`backend/evaluation/results/benchmark_dataset.jsonl`) are **FROZEN**.
>
> Pull Requests that modify benchmark datasets, tune scientific parameters, alter fusion formulas, or change evaluation metrics will be rejected to preserve peer-reviewed scientific reproducibility.

---

## 2. Development & PR Workflow

1. **Fork & Branch**: Create a feature branch from `main`:
   ```bash
   git checkout -b fix/issue-description
   ```
2. **Local Testing**: Run the full regression test suite before committing:
   ```bash
   cd backend
   PYTHONPATH=. venv/bin/pytest tests/ -v
   cd ../frontend
   npm run build
   ```
3. **Commit Conventions**: Use Conventional Commits (`fix:`, `feat:`, `docs:`, `perf:`).
4. **Code Quality**: Ensure zero TypeScript errors and adherence to existing code style.
5. **No Secrets**: Never commit `.env` files or API credentials.
