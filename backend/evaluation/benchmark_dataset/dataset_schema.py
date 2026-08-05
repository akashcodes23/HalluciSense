"""Phase 21.1 — Benchmark Dataset Schema & Exporter.

Defines the core BenchmarkExample data structure supporting 15 domains:
General Knowledge, Medicine, Law, Finance, History, Science, Computer Science,
Physics, Biology, Chemistry, News, Mathematics, Geography, Politics, Literature.

Includes JSONL, CSV, and Parquet serialization helpers.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

DOMAINS: List[str] = [
    "General Knowledge",
    "Medicine",
    "Law",
    "Finance",
    "History",
    "Science",
    "Computer Science",
    "Physics",
    "Biology",
    "Chemistry",
    "News",
    "Mathematics",
    "Geography",
    "Politics",
    "Literature",
]


@dataclass
class BenchmarkExample:
    """Standardized publication benchmark claim example."""

    id: str
    question: str
    response: str
    ground_truth: int  # 0 = Factual, 1 = Hallucinated
    domain: str
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "HaluEval/FactKG/Benchmark"
    llm_name: str = "GPT-4"
    label: str = "factual"  # "factual" or "hallucinated"
    claims: List[str] = field(default_factory=list)
    evidence_passages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.claims:
            self.claims = [self.response]
        self.label = "hallucinated" if self.ground_truth == 1 else "factual"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkDatasetManager:
    """Manages benchmark loading, domain filtering, and export."""

    def __init__(self, examples: Optional[List[BenchmarkExample]] = None):
        self.examples: List[BenchmarkExample] = examples or []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> BenchmarkExample:
        return self.examples[idx]

    def filter_by_domain(self, domain: str) -> BenchmarkDatasetManager:
        return BenchmarkDatasetManager([e for e in self.examples if e.domain.lower() == domain.lower()])

    def export_jsonl(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def export_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "question", "response", "ground_truth", "label", "domain", "difficulty", "source", "llm_name"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ex in self.examples:
                writer.writerow(ex.to_dict())

    @classmethod
    def load_jsonl(cls, path: Path) -> BenchmarkDatasetManager:
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                examples.append(
                    BenchmarkExample(
                        id=rec["id"],
                        question=rec.get("question", ""),
                        response=rec.get("response", ""),
                        ground_truth=int(rec.get("ground_truth", 0)),
                        domain=rec.get("domain", "General Knowledge"),
                        difficulty=rec.get("difficulty", "medium"),
                        source=rec.get("source", "Benchmark"),
                        llm_name=rec.get("llm_name", "GPT-4"),
                        claims=rec.get("claims", [rec.get("response", "")]),
                        evidence_passages=rec.get("evidence_passages", []),
                        metadata=rec.get("metadata", {}),
                    )
                )
        return cls(examples)
