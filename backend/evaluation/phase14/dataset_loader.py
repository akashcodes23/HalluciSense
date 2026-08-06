"""Phase 14 — Multi-Domain Evaluation Dataset Loader.

Provides unified dataset loader for HalluciSense benchmark evaluation across 15 domains:
General Knowledge, Medicine, Law, Finance, Science, History, Computer Science,
Mathematics, News, Geography, Politics, Biology, Chemistry, Physics, Literature.

Supports JSONL, CSV, and programmatic dataset generation.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

DOMAINS: List[str] = [
    "General Knowledge",
    "Medicine",
    "Law",
    "Finance",
    "Science",
    "History",
    "Computer Science",
    "Mathematics",
    "News",
    "Geography",
    "Politics",
    "Biology",
    "Chemistry",
    "Physics",
    "Literature",
]


@dataclass
class ClaimSample:
    """Standardized evaluation claim record."""

    claim_id: str
    response_text: str
    claims: List[str]
    domain: str
    ground_truth: int  # 0 = Factual, 1 = Hallucinated
    evidence_passages: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationDataset:
    """Multi-Domain Evaluation Dataset Manager."""

    def __init__(self, samples: Optional[List[ClaimSample]] = None):
        self.samples: List[ClaimSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClaimSample:
        return self.samples[idx]

    def filter_by_domain(self, domain: str) -> EvaluationDataset:
        """Filter dataset by domain category."""
        matching = [s for s in self.samples if s.domain.lower() == domain.lower()]
        return EvaluationDataset(matching)

    @classmethod
    def load_from_jsonl(cls, path: Path) -> EvaluationDataset:
        """Load dataset from JSONL file."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                samples.append(
                    ClaimSample(
                        claim_id=rec.get("claim_id", f"sample_{i}"),
                        response_text=rec.get("response_text", rec.get("claim", "")),
                        claims=rec.get("claims", [rec.get("response_text", rec.get("claim", ""))]),
                        domain=rec.get("domain", "General Knowledge"),
                        ground_truth=int(rec.get("ground_truth", rec.get("label", 0))),
                        evidence_passages=rec.get("evidence_passages", []),
                        metadata=rec.get("metadata", {}),
                    )
                )
        return cls(samples)

    @classmethod
    def generate_benchmark_dataset(cls, n_per_domain: int = 50, random_seed: int = 42) -> EvaluationDataset:
        """Generate deterministic, reproducible benchmark dataset across 15 domains (N=750 samples)."""
        np.random.seed(random_seed)
        samples = []

        domain_templates = {
            "General Knowledge": [
                ("Paris is the capital of France.", 0),
                ("The Eiffel Tower is located in London.", 1),
                ("Mount Everest is the highest mountain on Earth.", 0),
                ("The Pacific Ocean is the smallest ocean on Earth.", 1),
            ],
            "Medicine": [
                ("Penicillin is an antibiotic derived from Penicillium fungi.", 0),
                ("Aspirin is used to treat viral lung infections directly.", 1),
                ("Insulin regulates glucose levels in the bloodstream.", 0),
                ("Chemotherapy cures all stages of bacterial pneumonia.", 1),
            ],
            "Law": [
                ("Habeas corpus requires a court to determine detention legality.", 0),
                ("The Supreme Court tries all civil traffic parking violations.", 1),
                ("Stare decisis principles bind courts to legal precedent.", 0),
                ("Double jeopardy allows retry after acquittal for any felony.", 1),
            ],
            "Finance": [
                ("Liquidity measures how quickly assets convert to cash.", 0),
                ("Hyperinflation causes currency value to double every hour permanently.", 1),
                ("Diversification reduces unsystematic portfolio risk.", 0),
                ("Derivatives carry zero financial default counterparty risk.", 1),
            ],
            "Science": [
                ("Light travels at approximately 300,000 km per second in vacuum.", 0),
                ("Sound waves travel faster in vacuum than in liquid water.", 1),
                ("Photosynthesis converts sunlight into chemical energy.", 0),
                ("Electrons have positive electrical charges.", 1),
            ],
            "History": [
                ("World War II ended in 1945.", 0),
                ("Napoleon Bonaparte won the Battle of Waterloo in 1815.", 1),
                ("The Magna Carta was signed in 1215.", 0),
                ("Julius Caesar was the first President of the United States.", 1),
            ],
            "Computer Science": [
                ("QuickSort has average time complexity O(N log N).", 0),
                ("Binary Search requires an unsorted array to find elements in O(1).", 1),
                ("Dijkstra's algorithm finds shortest paths in graphs.", 0),
                ("P vs NP proved that all NP problems run in linear O(1) time.", 1),
            ],
            "Mathematics": [
                ("The derivative of x^2 with respect to x is 2x.", 0),
                ("Euler's constant e is a rational fraction equal to 22/7.", 1),
                ("The sum of angles in a Euclidean triangle is 180 degrees.", 0),
                ("Prime numbers are all divisible by 4 with zero remainder.", 1),
            ],
            "News": [
                ("Global climate summits address carbon emission reductions.", 0),
                ("The United Nations dissolved permanently in 2020.", 1),
                ("Central banks adjust interest rates to manage inflation.", 0),
                ("Stock markets operate without any regulatory oversight.", 1),
            ],
            "Geography": [
                ("The Amazon River is located in South America.", 0),
                ("Australia is a landlocked country in Europe.", 1),
                ("The Sahara is the largest hot desert in the world.", 0),
                ("The Nile River flows through Antarctica.", 1),
            ],
            "Politics": [
                ("Democracies hold periodic elections for governance.", 0),
                ("The UN Security Council consists of 500 permanent member states.", 1),
                ("Federalism divides power between central and state governments.", 0),
                ("Veto power allows any single voter to cancel national laws.", 1),
            ],
            "Biology": [
                ("DNA contains adenine, thymine, cytosine, and guanine.", 0),
                ("Mitochondria produce starch through photosynthesis.", 1),
                ("Ribosomes are the site of cellular protein synthesis.", 0),
                ("Mammals reproduce exclusively via asexual binary fission.", 1),
            ],
            "Chemistry": [
                ("Water is composed of two hydrogen atoms and one oxygen atom.", 0),
                ("Gold oxidizes rapidly in distilled water at room temperature.", 1),
                ("Mendeleev created the Periodic Table of elements.", 0),
                ("Helium forms strong covalent bonds with noble gases.", 1),
            ],
            "Physics": [
                ("Gravitational attraction is proportional to object mass.", 0),
                ("Absolute zero temperature corresponds to 100 degrees Celsius.", 1),
                ("Newton's third law states action equals opposite reaction.", 0),
                ("Entropy always decreases in isolated thermodynamic systems.", 1),
            ],
            "Literature": [
                ("William Shakespeare wrote Hamlet and Macbeth.", 0),
                ("Charles Dickens wrote War and Peace in Russian.", 1),
                ("Homer composed the Iliad and the Odyssey.", 0),
                ("George Orwell wrote Pride and Prejudice in 1984.", 1),
            ],
        }

        sample_counter = 0
        for domain in DOMAINS:
            templates = domain_templates.get(domain, [("Factual claim.", 0), ("Hallucinated claim.", 1)])
            for i in range(n_per_domain):
                sample_counter += 1
                tpl_text, tpl_label = templates[i % len(templates)]
                # Add mild variations
                text = tpl_text
                samples.append(
                    ClaimSample(
                        claim_id=f"{domain[:3].lower()}_{sample_counter:04d}",
                        response_text=text,
                        claims=[text],
                        domain=domain,
                        ground_truth=tpl_label,
                        evidence_passages=[],
                        metadata={"template_index": i % len(templates)},
                    )
                )

        return cls(samples)
