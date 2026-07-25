import pytest
import sys
from pathlib import Path

# Add app package directory to sys.path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

@pytest.fixture
def sample_evidence():
    from app.core.engine.types import EvidenceItem
    return [
        EvidenceItem(
            claim="Paris is the capital of France",
            snippet="Paris is the official capital and most populous city of France.",
            source_name="Wikipedia: Paris",
            source_url="https://en.wikipedia.org/wiki/Paris",
            similarity_score=0.95,
            is_supporting=True
        ),
        EvidenceItem(
            claim="The Eiffel Tower is in Rome",
            snippet="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            source_name="Wikipedia: Eiffel Tower",
            source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
            similarity_score=0.15,
            is_supporting=False
        )
    ]
