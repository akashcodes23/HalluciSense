"""HaluEval Dataset Adapter for HalluciSense Phase 6B.1.

Adapts HaluEval (RUCAIBox) cross-task records (QA, Dialogue, Summarization, General)
into canonical BenchmarkExample objects.
Generates paired factual (0) and hallucinated (1) examples with unique identifiers.
Preserves task configuration, knowledge/dialogue history, and original field values.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from evaluation.datasets.adapter import BenchmarkExample


class HaluEvalAdapter:
    """Adapter for processing HaluEval paired task datasets."""

    @staticmethod
    def process_records(
        records: List[Dict[str, Any]],
        task_name: str = "qa",
    ) -> Tuple[List[BenchmarkExample], Dict[str, Any]]:
        examples: List[BenchmarkExample] = []
        seen_ids = set()
        correct_count = 0
        hallucinated_count = 0

        for idx, row in enumerate(records, 1):
            knowledge = str(
                row.get("knowledge") or row.get("dialogue_history") or ""
            ).strip()
            question = str(
                row.get("question")
                or row.get("instruction")
                or row.get("document")
                or ""
            ).strip()

            if knowledge and question:
                prompt = f"Knowledge: {knowledge}\n\nQuestion: {question}"
            elif knowledge:
                prompt = f"Knowledge: {knowledge}"
            else:
                prompt = question or f"HaluEval Task {task_name.upper()} Sample {idx}"

            right_ans = str(
                row.get("right_answer")
                or row.get("right_response")
                or row.get("ground_truth")
                or ""
            ).strip()

            hallu_ans = str(
                row.get("hallucinated_answer")
                or row.get("hallucinated_response")
                or row.get("hallucinated_summary")
                or ""
            ).strip()

            base_id = str(row.get("id") or idx).strip()

            # 1. Correct response -> Label 0
            if right_ans:
                ex_id_corr = f"halueval:{task_name}:{base_id}:correct"
                if ex_id_corr in seen_ids:
                    ex_id_corr = f"halueval:{task_name}:{base_id}_{idx}:correct"
                seen_ids.add(ex_id_corr)

                ex_corr = BenchmarkExample(
                    example_id=ex_id_corr,
                    prompt=prompt,
                    response=right_ans,
                    label=0,
                    category=task_name.upper(),
                    metadata={
                        "dataset": "halueval",
                        "task": task_name,
                        "base_id": base_id,
                        "response_type": "correct",
                        "knowledge": knowledge,
                        "question": question,
                        "original_right_answer": right_ans,
                    },
                    synthetic_test_fixture=False,
                )
                examples.append(ex_corr)
                correct_count += 1

            # 2. Hallucinated response -> Label 1
            if hallu_ans:
                ex_id_hallu = f"halueval:{task_name}:{base_id}:hallucinated"
                if ex_id_hallu in seen_ids:
                    ex_id_hallu = f"halueval:{task_name}:{base_id}_{idx}:hallucinated"
                seen_ids.add(ex_id_hallu)

                ex_hallu = BenchmarkExample(
                    example_id=ex_id_hallu,
                    prompt=prompt,
                    response=hallu_ans,
                    label=1,
                    category=task_name.upper(),
                    metadata={
                        "dataset": "halueval",
                        "task": task_name,
                        "base_id": base_id,
                        "response_type": "hallucinated",
                        "knowledge": knowledge,
                        "question": question,
                        "original_hallucinated_answer": hallu_ans,
                    },
                    synthetic_test_fixture=False,
                )
                examples.append(ex_hallu)
                hallucinated_count += 1

        stats = {
            "task": task_name,
            "raw_records": len(records),
            "processed_examples": len(examples),
            "correct_pair_examples": correct_count,
            "hallucinated_pair_examples": hallucinated_count,
        }

        return examples, stats

    @staticmethod
    def load_from_json(
        file_path: Union[str, Path], task_name: str = "qa"
    ) -> Tuple[List[BenchmarkExample], Dict[str, Any]]:
        path = Path(file_path)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip():
                        records.append(json.loads(line))

        return HaluEvalAdapter.process_records(records, task_name=task_name)
