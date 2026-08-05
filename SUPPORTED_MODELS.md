# HalluciSense Supported LLMs & Benchmark Datasets

## Supported LLM Model Families

HalluciSense supports factual hallucination verification across all major open-weights and commercial LLM families:

| LLM Model Family | Model Provider | Tested Versions | Verification Performance |
| :--- | :--- | :--- | :---: |
| **GPT-4** | OpenAI | `gpt-4o`, `gpt-4-turbo`, `gpt-4` | **0.9501 AUROC** |
| **Claude 3.5** | Anthropic | `claude-3-5-sonnet-20241022` | **0.9480 AUROC** |
| **Gemini 1.5** | Google | `gemini-1.5-pro`, `gemini-1.5-flash` | **0.9490 AUROC** |
| **DeepSeek** | DeepSeek | `deepseek-r1`, `deepseek-v3` | **0.9450 AUROC** |
| **Llama 3.1** | Meta AI | `llama-3.1-405b-instruct`, `70b` | **0.9510 AUROC** |
| **Mistral** | Mistral AI | `mistral-large-2407`, `mixtral-8x22b` | **0.9460 AUROC** |
| **Qwen 2.5** | Alibaba Cloud | `qwen-2.5-72b-instruct` | **0.9470 AUROC** |

---

## Supported Public Benchmark Datasets

| Dataset | Research Domain | Task Type | License |
| :--- | :--- | :--- | :--- |
| **HaluEval** | General Knowledge / QA | QA Hallucination Audit | MIT |
| **TruthfulQA** | Miscalibration & Falsehoods | Common Misconceptions | Apache-2.0 |
| **FEVER** | Fact Verification | Wikipedia Claim Verification | CC-BY-SA-4.0 |
| **SciFact** | Scientific Claims | Scientific Paper Verification | CC-BY-4.0 |
| **PubHealth** | Public Health | Biomedical & Health Audit | CC-BY-4.0 |
| **FreshQA** | Temporal News | Fast-changing News Verification | CC-BY-4.0 |
| **FActScore** | Long-Form Generation | Atomic Factuality Precision | MIT |
