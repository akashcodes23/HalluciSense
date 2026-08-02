"""HaluBench Dataset Adapter for HalluciSense Phase 6B.1.

Adapts HaluBench (Patronus AI / Vectara) raw records into canonical BenchmarkExample objects.
Maps 'PASS' to 0 (factual) and 'FAIL' to 1 (hallucinated).
Preserves rich metadata including passage, question, source_ds, and original ID.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from evaluation.datasets.adapter import BenchmarkExample, BenchmarkDataset, BenchmarkAdapter


HALUBENCH_LABEL_MAP: Dict[str, int] = {
    "PASS": 0,
    "pass": 0,
    "0": 0,
    0: 0,
    "FAIL": 1,
    "fail": 1,
    "1": 1,
    1: 1,
}


class HaluBenchAdapter:
    """Adapter for processing and normalizing HaluBench dataset records."""

    @staticmethod
    def process_records(
        records: List[Dict[str, Any]],
    ) -> List[BenchmarkExample]:
        examples: List[BenchmarkExample] = []
        seen_ids = set()

        for idx, row in enumerate(records, 1):
            raw_id = str(row.get("id") or f"row_{idx}").strip()
            ex_id = f"halubench:{raw_id}"
            if ex_id in seen_ids:
                ex_id = f"halubench:{raw_id}_{idx}"
            seen_ids.add(ex_id)

            passage = str(row.get("passage", "")).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            raw_label = row.get("label")
            source_ds = str(row.get("source_ds", "unknown")).strip()

            if raw_label not in HALUBENCH_LABEL_MAP:
                raise ValueError(
                    f"Unmapped HaluBench label '{raw_label}' at record {idx} (id={raw_id})."
                )
            label = HALUBENCH_LABEL_MAP[raw_label]

            prompt = (
                f"Context: {passage}\n\nQuestion: {question}"
                if passage
                else question
            )

            meta = {
                "dataset": "halubench",
                "source_ds": source_ds,
                "original_id": raw_id,
                "passage": passage,
                "question": question,
                "original_label": raw_label,
            }

            example = BenchmarkExample(
                example_id=ex_id,
                prompt=prompt,
                response=answer,
                label=label,
                category=source_ds.upper(),
                metadata=meta,
                synthetic_test_fixture=False,
            )
            examples.append(example)

        return examples

    @staticmethod
    def load_from_parquet(file_path: Union[str, Path]) -> List[BenchmarkExample]:
        path = Path(file_path)
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        return HaluBenchAdapter.process_records(records)
