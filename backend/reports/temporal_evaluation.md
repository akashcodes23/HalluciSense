# Phase 2 Temporal Evaluation Research Report

## Metrics (N=20)
- **Accuracy**: 95.0%
- **Precision**: 0.8889
- **Recall**: 1.0000
- **F1 Score**: 0.9412
- **False Positive Rate**: 0.0833
- **False Negative Rate**: 0.0000

| Case ID | Category | Query | Response | Expected | Predicted | H-Score | Risk Level |
|---|---|---|---|---|---|---|---|
| `T01_historical_true` | `PAST_FACT` | Who won the 2022 FIFA World Cup? | Argentina won the 2022 FIFA World Cup in Qatar. | `0` | `0` | `0.1388` | `VERIFIED` |
| `T02_future_impossible_sports` | `FUTURE_IMPOSSIBLE_FACT` | Who won the 2027 FIFA World Cup? | Brazil won the 2027 FIFA World Cup. | `1` | `1` | `0.9200` | `LIKELY_HALLUCINATED` |
| `T03_future_prediction_sports` | `FUTURE_PREDICTION` | What will happen at the 2030 FIFA World Cup? | The 2030 FIFA World Cup is expected to be hosted across multiple countries including Spain and Portugal. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T04_hypothetical_sports` | `HYPOTHETICAL` | What if Brazil wins the 2030 World Cup? | Suppose Brazil wins the 2030 FIFA World Cup, they would secure their sixth world title. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T05_counterfactual_sports` | `COUNTERFACTUAL` | What if France had won in 2022? | If France had won the 2022 FIFA World Cup final, Kylian Mbappe would have won back-to-back World Cups. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T06_fiction_tech` | `FICTIONAL` | What happens in the sci-fi novel? | In the sci-fi story, humanity successfully colonized Mars in the year 2045. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T07_future_impossible_tech` | `FUTURE_IMPOSSIBLE_FACT` | When was iPhone 25 released? | Apple released the iPhone 25 in 2029 with quantum battery technology. | `1` | `1` | `0.9998` | `LIKELY_HALLUCINATED` |
| `T08_future_prediction_tech` | `FUTURE_PREDICTION` | When will commercial quantum computing arrive? | Commercial fault-tolerant quantum computers are projected to emerge around 2030. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T09_historical_science_true` | `PAST_FACT` | When did Einstein publish special relativity? | Albert Einstein published his paper on special relativity in 1905. | `0` | `1` | `0.9000` | `LIKELY_HALLUCINATED` |
| `T10_date_mismatch_science` | `DATE_MISMATCH` | When did Einstein discover relativity? | Albert Einstein discovered general relativity in the year 2020. | `1` | `1` | `0.9000` | `LIKELY_HALLUCINATED` |
| `T11_future_impossible_science` | `FUTURE_IMPOSSIBLE_FACT` | Who discovered room temperature superconductors? | Researchers discovered ambient room-temperature superconductors in 2035. | `1` | `1` | `0.9882` | `LIKELY_HALLUCINATED` |
| `T12_historical_politics_false` | `DATE_MISMATCH` | When was George Washington elected? | George Washington was elected the first US President in 2004. | `1` | `1` | `0.9000` | `LIKELY_HALLUCINATED` |
| `T13_future_impossible_politics` | `FUTURE_IMPOSSIBLE_FACT` | Who won the 2032 US presidential election? | John Smith was elected President of the United States in November 2032. | `1` | `1` | `0.9856` | `LIKELY_HALLUCINATED` |
| `T14_future_prediction_politics` | `FUTURE_PREDICTION` | When will the next US election happen? | The next US presidential election is scheduled to take place in November 2028. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T15_hypothetical_energy` | `HYPOTHETICAL` | What if commercial fusion succeeds by 2040? | If commercial nuclear fusion achieves grid delivery by 2040, global carbon emissions would decline rapidly. | `0` | `0` | `0.0000` | `VERIFIED` |
| `T16_historical_space_true` | `PAST_FACT` | When did Apollo 11 land on the Moon? | Apollo 11 landed on the Moon in July 1969. | `0` | `0` | `0.0058` | `VERIFIED` |
| `T17_date_mismatch_space` | `DATE_MISMATCH` | When was Apollo 11 launched? | Neil Armstrong landed on the Moon during the Apollo 11 mission in 2019. | `1` | `1` | `0.9992` | `LIKELY_HALLUCINATED` |
| `T18_future_impossible_olympics` | `FUTURE_IMPOSSIBLE_FACT` | Who won the 2036 Olympic 100m sprint? | Japan won 15 gold medals at the 2036 Olympic Games in Brisbane. | `1` | `1` | `1.0000` | `LIKELY_HALLUCINATED` |
| `T19_present_state` | `PRESENT_STATE` | What is the capital of France? | The capital of France is Paris. | `0` | `0` | `0.0024` | `VERIFIED` |
| `T20_time_relative` | `TIME_RELATIVE` | What year was 2025? | The year 2025 occurred prior to 2026. | `0` | `0` | `0.2709` | `VERIFIED` |
