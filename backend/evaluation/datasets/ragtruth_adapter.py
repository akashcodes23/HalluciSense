"""RAGTruth Dataset Adapter for HalluciSense Phase 6B.1.

Adapts RAGTruth (ParticleMedia) raw response and source records into canonical BenchmarkExample objects.
Derives response-level binary label (0 = no hallucination spans, 1 = >= 1 hallucination span).
Validates span bounds (0 <= start < end <= len(response)) and text alignment.
Preserves fine-grained span annotations in metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from evaluation.datasets.adapter import BenchmarkExample


class RAGTruthAdapter:
    """Adapter for processing and normalizing RAGTruth response and span annotations."""

    @staticmethod
    def validate_span_bounds(
        response_text: str, span: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates span bounds: 0 <= start < end <= len(response_text).
        Checks character alignment if text is present.
        Returns (is_valid, warning_message).
        """
        start = span.get("start")
        end = span.get("end")
        text = span.get("text", "")

        if start is None or end is None:
            return False, "Span missing 'start' or 'end' position."

        if not (isinstance(start, int) and isinstance(end, int)):
            return False, f"Span positions must be integers, got start={start}, end={end}."

        if start < 0 or end <= start or end > len(response_text):
            return False, f"Out-of-bounds span [{start}:{end}] for response of length {len(response_text)}."

        # Verify text alignment if present
        actual_slice = response_text[start:end]
        if text and actual_slice != text:
            return True, f"Span text mismatch: expected '{text}', slice was '{actual_slice}'."

        return True, None

    @staticmethod
    def process_records(
        responses: List[Dict[str, Any]],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[BenchmarkExample], Dict[str, Any]]:
        source_map = {}
        if sources:
            for src in sources:
                src_id = str(src.get("source_id", "")).strip()
                if src_id:
                    source_map[src_id] = src

        examples: List[BenchmarkExample] = []
        seen_ids = set()
        total_spans = 0
        responses_with_spans = 0
        span_type_counts: Dict[str, int] = {}
        warnings_list: List[str] = []

        for idx, resp in enumerate(responses, 1):
            raw_id = str(resp.get("id") or f"resp_{idx}").strip()
            ex_id = f"ragtruth:{raw_id}"
            if ex_id in seen_ids:
                ex_id = f"ragtruth:{raw_id}_{idx}"
            seen_ids.add(ex_id)

            resp_text = str(resp.get("response", "")).strip()
            if not resp_text:
                continue

            src_id = str(resp.get("source_id", "")).strip()
            src_info = source_map.get(src_id, {})
            prompt = str(src_info.get("prompt") or src_info.get("source_info") or "").strip()
            if not prompt:
                prompt = f"Source ID: {src_id}"

            raw_labels = resp.get("labels", [])
            valid_spans = []

            for span in raw_labels:
                is_valid, warn = RAGTruthAdapter.validate_span_bounds(resp_text, span)
                if is_valid:
                    valid_spans.append(span)
                    total_spans += 1
                    lbl_type = str(span.get("label_type", "hallucination")).strip()
                    span_type_counts[lbl_type] = span_type_counts.get(lbl_type, 0) + 1
                if warn:
                    warnings_list.append(f"Response ID {raw_id}: {warn}")

            if len(valid_spans) > 0:
                responses_with_spans += 1

            # Response-level binary label: 0 if no hallucination spans, 1 if >= 1
            label = 1 if len(valid_spans) > 0 else 0

            meta = {
                "dataset": "ragtruth",
                "source_id": src_id,
                "model": str(resp.get("model", "unknown")),
                "temperature": resp.get("temperature"),
                "task_type": str(src_info.get("task_type", "unknown")),
                "source": str(src_info.get("source", "unknown")),
                "hallucination_spans": valid_spans,
                "split": str(resp.get("split", "test")),
                "original_labels": raw_labels,
            }

            example = BenchmarkExample(
                example_id=ex_id,
                prompt=prompt,
                response=resp_text,
                label=label,
                category=meta["task_type"].upper(),
                metadata=meta,
                synthetic_test_fixture=False,
            )
            examples.append(example)

        span_stats = {
            "total_responses": len(examples),
            "responses_with_hallucination_spans": responses_with_spans,
            "total_hallucination_spans": total_spans,
            "span_type_distribution": span_type_counts,
            "validation_warnings_count": len(warnings_list),
            "sample_warnings": warnings_list[:5],
        }

        return examples, span_stats

    @staticmethod
    def load_from_jsonl(
        response_file: Union[str, Path],
        source_info_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[BenchmarkExample], Dict[str, Any]]:
        responses = []
        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))

        sources = None
        if source_info_file and Path(source_info_file).exists():
            sources = []
            with open(source_info_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sources.append(json.loads(line))

        return RAGTruthAdapter.process_records(responses, sources)
