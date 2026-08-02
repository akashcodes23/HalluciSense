# HalluciSense Limitations & Distribution Shift Notes

1. **Pillar-2 Distribution Shift**: Cross-encoder NLI scores on unseen datasets may shift downward ($\text{SMD} = -0.8481$), causing probability output compression.
2. **Threshold Sensitivity**: The operating threshold $\tau^* = 0.54$ tuned on DEV may require conformal recalibration on highly out-of-domain evaluation splits.
3. **NLI Language Support**: Current NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) is primarily evaluated on English language texts.
