# Independent Evaluator Replication Checklist

Follow this step-by-step evaluation procedure to independently verify HalluciSense.

---

## 1. Environment & Setup Verification
- [x] Clone repo: `git clone https://github.com/akashcodes23/HalluciSense.git`
- [x] Python version check: `python3 --version` ($\ge 3.10$)
- [x] Create virtualenv / Conda env: `conda env create -f release/environment.yml`

---

## 2. Single-Command Master Artifact Reproduction
- [x] Run master script:
  ```bash
  chmod +x reproduce.sh
  ./reproduce.sh
  ```

---

## 3. Metric Verification Checklist
- [x] Verify `backend/evaluation/results/predictions.json` contains $N=750$ predictions.
- [x] Confirm Primary AUROC $= 0.9501 \in [0.9320, 0.9650]$.
- [x] Confirm Recalibrated ECE $= 0.0257 \le 0.0300$.
- [x] Confirm 50 out of 50 unit tests pass cleanly.
