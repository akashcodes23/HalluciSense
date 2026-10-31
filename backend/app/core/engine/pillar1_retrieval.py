import re
from typing import List, Tuple, Optional, Dict, Any, Set

import httpx

from .types import Pillar1Result, EvidenceItem
from .entailment import EvidenceEntailmentEngine
from .temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality
from .numeric_unit_checker import NumericUnitChecker, NumericUnitStatus
from .negation_detector import NegationDetector
from .causal_direction import CausalDirectionChecker
from app.core.verification.symbolic_verifier import evaluate_arithmetic_claim
from app.core.verification.unit_verifier import evaluate_unit_claim
from app.core.engine.scientific_claim_reasoner import ScientificClaimReasoner, ClaimSRO
from app.core.verification.precision_policy import NumericalPrecisionPolicy, PrecisionDomain

GENERIC_GEOPOLITICAL_STOP_WORDS = {
    "state", "the state", "country", "the country", "nation", "the nation",
    "city", "the city", "province", "the province", "region", "the region",
    "territory", "the territory", "place", "area", "it", "this", "that", "these", "those",
    "the", "a", "an", "terms", "terms of", "its", "their", "capital", "capitals", "headquarters",
    "seat", "government", "administration", "authority",
    "what", "which", "who", "whom", "whose", "where", "why", "how", "when",
    "question", "questions", "answer", "answers", "example", "examples", "statement", "query"
}


def is_valid_geopolitical_entity(name: str) -> bool:
    """Validate that an extracted name is a genuine geopolitical entity, not a grammatical placeholder."""
    if not name:
        return False
    clean = re.sub(r'^(?:the|a|an)\s+', '', name.strip(), flags=re.IGNORECASE).strip().lower()
    if clean in GENERIC_GEOPOLITICAL_STOP_WORDS:
        return False
    if len(clean) < 3:
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    words = clean.split()
    if len(words) > 4:
        return False
    # Reject strings containing grammatical conjunctions, relative pronouns, or prepositions
    if any(w in {"is", "was", "are", "were", "and", "or", "which", "that", "with", "from", "for", "in", "on", "at"} for w in words):
        return False
    return True


def _clean_entity_mention(name: str) -> str:
    """Strip leading conjunctions, articles, and clause markers from extracted entity names."""
    if not name:
        return ""
    clean = name.strip().rstrip(",.")
    m = re.search(r'\b(?:and|or|but|while|whereas|that)\s+([A-Za-z\s’\'-]+)$', clean, re.IGNORECASE)
    if m:
        clean = m.group(1).strip()
    clean = re.sub(r'^(?:the|a|an|city of)\s+', '', clean, flags=re.IGNORECASE).strip()
    return clean


def extract_capital_relation(text: str) -> Optional[Tuple[str, str]]:
    """Extract (city, state/country) tuple from capital assertions.

    Returns (city, entity) in lowercase if text asserts a valid capital relation, else None.
    Handles variations like:
    - 'The capital of Germany is Berlin.'
    - 'The capital of Germany is the city of Berlin.'
    - 'Berlin is the capital of Germany.'
    - 'Bengaluru, the capital of Karnataka, ...'
    - 'The capital city of Australia is Canberra.'
    - 'Germany’s capital is Berlin.'
    - 'The economy of Bengaluru, the state capital of Karnataka, is ...'

    Employs clause-level negation handling to preserve affirmative assertions in compound sentences:
    - 'Berlin is not the capital of France.' -> None
    - 'Berlin is the capital of Germany.' -> ('berlin', 'germany')
    - 'Berlin is the capital of Germany, but Paris is not.' -> ('berlin', 'germany')
    - 'Paris is not the capital of Germany. Berlin is.' -> ('berlin', 'germany')
    - 'The capital of Germany is Berlin, not Munich.' -> ('berlin', 'germany')

    Strictly rejects generic placeholders and interrogatives.
    """
    if not text:
        return None

    # Process sentence by sentence, tracking negative context for elliptical assertions
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
    negative_context_entity: Optional[str] = None

    for sent in sentences:
        # Check if sentence is an elliptical affirmative referring to previous negative claim
        # E.g. 'Paris is not the capital of Germany. Berlin is.'
        if negative_context_entity:
            m_ellip = re.match(r'^([A-Za-z\s’\'-]{2,35}?)\s+(?:is|was)\.?$', sent, re.IGNORECASE)
            if m_ellip:
                cand_city = _clean_entity_mention(m_ellip.group(1))
                if is_valid_geopolitical_entity(cand_city):
                    return cand_city.lower(), negative_context_entity.lower()

        # Check for explicit negative capital assertion to populate context for follow-up
        m_neg_search = re.search(
            r'\b([A-Za-z\s’\'-]{2,35}?)\s+(?:is|was)\s+not\s+(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?of\s+(?:the\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
            sent,
            re.IGNORECASE,
        )
        if m_neg_search:
            neg_ent = _clean_entity_mention(m_neg_search.group(2))
            if is_valid_geopolitical_entity(neg_ent):
                negative_context_entity = neg_ent

        # Segment sentence into clauses on contrastive conjunctions, semicolons, and commas before contrast tags
        # E.g. 'Berlin is the capital of Germany, but Paris is not.' -> ['Berlin is the capital of Germany', 'Paris is not']
        # E.g. 'The capital of Germany is Berlin, not Munich.' -> ['The capital of Germany is Berlin', 'not Munich']
        clauses = re.split(
            r'\s*[,;]\s*(?:but|however|although|though|whereas|while|yet|not\b)\s*|\s+(?:but|whereas|while)\s+',
            sent,
            flags=re.IGNORECASE,
        )

        for clause in clauses:
            clause = clause.strip()
            if not clause or "capital" not in clause.lower():
                continue

            # Check if this specific clause has internal negation on the capital relation
            # E.g. 'Berlin is not the capital of France', 'is not capital', 'not the capital'
            if re.search(r'\b(?:is|was|are|were)\s+not\b|\bnot\s+(?:the\s+)?capital\b|\bnever\s+the\s+capital\b', clause, re.IGNORECASE):
                continue

            # Pattern 2: (The) (modifier) capital (city) of Y is/was (the) (city of) X
            # E.g. 'The capital of Germany is Berlin', 'The capital city of Australia is Canberra',
            #      'The capital of Germany is the city of Berlin.'
            m2 = re.search(
                r'\b(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?of\s+(?:the\s+)?(?:(?:[a-z]{3,15}\s+){0,2}(?:state|country|nation|province|territory)\s+of\s+)?([A-Za-z\s’\'-]{2,35}?)\s+(?:is|was)\s+(?:the\s+)?(?:city\s+of\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
                clause,
                re.IGNORECASE,
            )
            if m2:
                entity = _clean_entity_mention(m2.group(1))
                city = _clean_entity_mention(m2.group(2))
                city = re.split(r'\b(?:officially|formerly|known as)\b', city, flags=re.IGNORECASE)[0].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

            # Pattern 3: Y's (modifier) capital (city) is/was (the) (city of) X
            # E.g. "Germany's capital is Berlin", "Karnataka's state capital is Bengaluru"
            m3 = re.search(
                r'\b([A-Za-z\s’\'-]{2,35}?)(?:\'s|’s)\s+(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?(?:is|was)\s+(?:the\s+)?(?:city\s+of\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
                clause,
                re.IGNORECASE,
            )
            if m3:
                entity = _clean_entity_mention(m3.group(1))
                city = _clean_entity_mention(m3.group(2))
                city = re.split(r'\b(?:officially|formerly|known as)\b', city, flags=re.IGNORECASE)[0].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

            # Pattern 1: X is/was (the) (official/state/...) capital (city) of Y
            # E.g. 'Berlin is the capital of Germany.', 'Bengaluru is the official state capital of Karnataka.'
            # Also supports parenthetical annotations like 'Canberra (KAN-brə) is the capital city of Australia'
            m1 = re.search(
                r'\b([A-Za-z\s’\'-]{2,35}?)(?:\s*\([^)]*\))?(?:,\s*(?:also known as|officially|formerly)[^,]+?,?)?\s+(?:is|was)\s+(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?(?:and\s+[a-z\s]{1,20}?\s+city\s+)?of\s+(?:the\s+)?(?:(?:[a-z]{3,15}\s+){0,2}(?:state|country|nation|province|territory)\s+of\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
                clause,
                re.IGNORECASE,
            )
            if m1:
                city = _clean_entity_mention(m1.group(1))
                entity = _clean_entity_mention(m1.group(2))
                city = re.split(r'\b(?:officially|formerly|known as)\b', city, flags=re.IGNORECASE)[0].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

            # Pattern 5: X, (the) (state/national) capital of Y (city apposition first)
            # E.g. 'Bengaluru, the capital of Karnataka, is...', 'The economy of Bengaluru, the state capital of Karnataka, ...'
            m5 = re.search(
                r'\b([A-Za-z\s’\'-]{2,35}?)\s*,\s+(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?of\s+(?:the\s+)?(?:(?:[a-z]{3,15}\s+){0,2}(?:state|country|nation|province|territory)\s+of\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
                clause,
                re.IGNORECASE,
            )
            if m5:
                city = m5.group(1).strip().rstrip(",.")
                entity = _clean_entity_mention(m5.group(2))
                city = re.split(r'\b(?:officially|formerly|known as|of)\b', city, flags=re.IGNORECASE)[-1].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

            # Pattern 4: (The) capital of Y, X, is/was ... (capital apposition first)
            # E.g. 'The capital of Karnataka, Bengaluru, is known as...'
            # Strictly rejects generic 'state' because 'state' fails is_valid_geopolitical_entity.
            m4 = re.search(
                r'\b(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?of\s+(?:the\s+)?([A-Za-z\s’\'-]{2,35}?)\s*,\s*([A-Za-z\s’\'-]{2,35}?)\s*,\s*(?:is|was|\b)',
                clause,
                re.IGNORECASE,
            )
            if m4:
                entity = _clean_entity_mention(m4.group(1))
                city = _clean_entity_mention(m4.group(2))
                city = re.split(r'\b(?:officially|formerly|known as|and|in|on|at|while|after|before)\b', city, flags=re.IGNORECASE)[0].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

            # Pattern 6: in/of Y, the capital X
            # E.g. 'In Germany, the capital Berlin has...'
            m6 = re.search(
                r'\b(?:of|in)\s+([A-Za-z\s’\'-]{2,35}?)\s*,\s*(?:the\s+)?(?:(?:official|state|national|federal|provincial|administrative)\s+)*capital\s+(?:city\s+)?([A-Za-z\s’\'-]{2,35}?)(?:\.|\s*[,;]|\s+and\b|\s+with\b|\s+in\b|\s+as\b|\Z)',
                clause,
                re.IGNORECASE,
            )
            if m6:
                entity = _clean_entity_mention(m6.group(1))
                city = _clean_entity_mention(m6.group(2))
                city = re.split(r'\b(?:officially|formerly|known as|and|in|on|at|while|after|before)\b', city, flags=re.IGNORECASE)[0].strip()
                city = _clean_entity_mention(city)
                if is_valid_geopolitical_entity(city) and is_valid_geopolitical_entity(entity):
                    return city.lower(), entity.lower()

    return None


STOP_FORMULA_WORDS = {"is", "are", "was", "were", "can", "could", "of", "for", "a", "an", "the", "any", "which", "to", "in", "by", "that", "be", "as", "such", "several", "its", "each"}

ELEMENT_MAP = {
    "water": {"hydrogen", "oxygen"},
    "h2o": {"hydrogen", "oxygen"},
    "carbon dioxide": {"carbon", "oxygen"},
    "co2": {"carbon", "oxygen"},
    "carbon monoxide": {"carbon", "oxygen"},
    "methane": {"carbon", "hydrogen"},
    "ch4": {"carbon", "hydrogen"},
    "ammonia": {"nitrogen", "hydrogen"},
    "nh3": {"nitrogen", "hydrogen"},
    "table salt": {"sodium", "chlorine"},
    "sodium chloride": {"sodium", "chlorine"},
    "nacl": {"sodium", "chlorine"},
    "ozone": {"oxygen"},
    "o3": {"oxygen"},
    "hydrogen peroxide": {"hydrogen", "oxygen"},
    "h2o2": {"hydrogen", "oxygen"},
}


def extract_chemical_composition(text: str) -> Optional[Tuple[str, Set[str]]]:
    """Extract (substance, elements) from chemical composition assertions."""
    m = re.search(
        r"\b([a-zA-Z\s]{3,20}?)\s+(?:is\s+)?(?:composed\s+of|consists?\s+of|made\s+up\s+of)\s+([a-zA-Z\s,and]+?)(?:\.|\s*[,;]|\Z)",
        text,
        re.IGNORECASE,
    )
    if m:
        subj = m.group(1).strip().lower()
        raw_elems = m.group(2).lower()
        elems = set(re.findall(r"\b[a-z]{3,15}\b", raw_elems)) - {"and", "of", "the", "both", "all", "mainly", "primarily"}
        if subj in ELEMENT_MAP and elems:
            return subj, elems
    return None


def extract_chemical_formula(text: str) -> Optional[Tuple[str, str]]:
    """Extract (subject, chemical_formula) from text if present.

    Handles statements like:
    - 'Water has the chemical formula H2O'
    - 'Water is an inorganic compound with the chemical formula H2O'
    - 'Methane (CH4) is a hydrocarbon'
    Excludes English stop-words from being treated as chemical symbols.
    """
    m = re.search(r'\b([A-Za-z]{3,})\b[^.!?]*?\b(?:chemical\s+formula|formula)\s*(?:is|:)?\s*([A-Za-z0-9]+)\b', text, re.IGNORECASE)
    if m:
        subj = m.group(1).lower()
        form = m.group(2)
        if form.lower() not in STOP_FORMULA_WORDS and len(form) >= 2:
            return subj, form.upper()
    m2 = re.search(r'\b([A-Za-z]{3,})\s*\(([A-Za-z0-9]{2,10})\)', text)
    if m2:
        subj = m2.group(1).lower()
        form = m2.group(2)
        if form.lower() not in STOP_FORMULA_WORDS:
            return subj, form.upper()
    return None


NON_ENTITY_BASES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "day", "month", "year", "grade", "page", "section", "part",
    "chapter", "table", "figure", "rule", "step", "level"
}


def has_conflicting_designated_entity(claim: str, snippet: str, source_name: str = "") -> bool:
    """Detect if an evidence snippet/source primarily describes a distinct sibling entity in the same series.

    Prevents false cross-encoder contradictions where an unrelated premise (e.g., Apollo 12)
    is erroneously evaluated as contradicting a claim about a distinct entity (e.g., Apollo 11).
    """
    clean_claim = re.sub(r"^(?:the\s+)", "", claim.strip(), flags=re.IGNORECASE)
    m_claim = re.match(r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+|[IVXLCDM]+)\b", clean_claim)
    if not m_claim:
        m_claim_all = re.findall(r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+|[IVXLCDM]+)\b", claim)
        valid = [(b.lower(), d.lower()) for b, d in m_claim_all if b.lower() not in NON_ENTITY_BASES]
        if len(valid) == 1:
            c_base, c_des = valid[0]
        else:
            return False
    else:
        c_base, c_des = m_claim.group(1).lower(), m_claim.group(2).lower()

    if c_base in NON_ENTITY_BASES:
        return False

    check_texts = [source_name, snippet[:120]]
    for t in check_texts:
        clean_t = re.sub(r"^(?:wikipedia:\s*|the\s+)", "", t.strip(), flags=re.IGNORECASE)
        m_ev = re.match(r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+|[IVXLCDM]+)\b", clean_t)
        if m_ev:
            e_base, e_des = m_ev.group(1).lower(), m_ev.group(2).lower()
            if e_base == c_base and e_des != c_des:
                return True
    return False


class EventTemporalAnchorResolver:
    """Resolve event/entity time spans dynamically through Wikidata.

    No entity names or dates are hardcoded. The resolver is deliberately
    conservative: it only activates for relative temporal statements without
    explicit four-digit years and returns a contradiction only when two
    independently resolved anchors make the stated relation impossible.
    """

    API_URL = "https://www.wikidata.org/w/api.php"
    TIME_PROPERTIES = ("P585", "P580", "P582", "P571", "P576", "P575")
    MAX_ANCHORS = 2
    TIMEOUT_SECONDS = 0.8

    def __init__(self):
        from collections import OrderedDict
        self.MAX_CACHE_ENTRIES = 512
        self._cache: OrderedDict = OrderedDict()
        self.last_lookup_count = 0

    @staticmethod
    def _extract_year(value: Any) -> Optional[int]:
        if not isinstance(value, str):
            return None
        match = re.search(r"[+-](\d{4})-", value)
        return int(match.group(1)) if match else None

    def _normalise_candidate(self, raw: str) -> str:
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def extract_anchor_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []

        # 1. Capitalized multi-word named entities.
        for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9'’-]+(?:\s+[A-Z][A-Za-z0-9'’-]+){1,5})\b", text):
            candidate = self._normalise_candidate(match.group(0))
            if candidate and len(candidate.split()) <= 5 and candidate.lower() not in {c.lower() for c in candidates}:
                candidates.append(candidate)

        # 2. Relational subject/object captures.
        relation_object = re.compile(
            r"\b(?:prior to|earlier than|later than|preceding|following|before|after|since|during)\s+([A-Za-z0-9'’\-\s]{3,60})",
            re.IGNORECASE,
        )
        for match in relation_object.finditer(text):
            candidate = self._normalise_candidate(match.group(1))
            candidate = re.split(r"\b(?:was|were|is|are|happened|occurred|began|ended)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if candidate and len(candidate.split()) <= 5 and candidate.lower() not in {c.lower() for c in candidates}:
                candidates.append(candidate)

        return candidates[: self.MAX_ANCHORS]

    def _search_entity(self, label: str) -> Optional[str]:
        if label in self._cache:
            cached = self._cache[label]
            return cached.get("id") if cached else None

        try:
            self.last_lookup_count += 1
            response = self.client.get(
                self.API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": label,
                    "language": "en",
                    "type": "item",
                    "limit": 1,
                    "format": "json",
                },
                headers={"User-Agent": "HalluciSense/Phase6 temporal-research"},
            )
            response.raise_for_status()
            results = response.json().get("search", [])
            if not results:
                if len(self._cache) >= self.MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
                self._cache[label] = None
                return None
            entity_id = results[0].get("id")
            if len(self._cache) >= self.MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
            self._cache[label] = {"id": entity_id} if entity_id else None
            return entity_id
        except Exception:
            if len(self._cache) >= self.MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
            self._cache[label] = None
            return None

    def _fetch_time_span(self, entity_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.get(
                self.API_URL,
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "claims|labels|descriptions",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": "HalluciSense/Phase6 temporal-research"},
            )
            response.raise_for_status()
            entity = response.json().get("entities", {}).get(entity_id, {})
            claims = entity.get("claims", {})
            years: List[int] = []
            for prop in self.TIME_PROPERTIES:
                for statement in claims.get(prop, []):
                    value = (
                        statement.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                    )
                    year = self._extract_year(value.get("time") if isinstance(value, dict) else None)
                    if year is not None:
                        years.append(year)

            if not years:
                return None

            return {
                "id": entity_id,
                "label": entity.get("labels", {}).get("en", {}).get("value", ""),
                "start": min(years),
                "end": max(years),
                "points": sorted(set(years)),
            }
        except Exception:
            return None

    def resolve(self, text: str) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        for candidate in self.extract_anchor_candidates(text):
            entity_id = self._search_entity(candidate)
            if not entity_id:
                continue
            span = self._fetch_time_span(entity_id)
            if span:
                span["query"] = candidate
                anchors.append(span)
            if len(anchors) >= self.MAX_ANCHORS:
                break
        return anchors

    @staticmethod
    def _relation(text: str) -> Optional[str]:
        lower = text.lower()
        for relation in ("prior to", "earlier than", "later than", "preceding", "following", "before", "after", "since", "during"):
            if re.search(rf"\b{re.escape(relation)}\b", lower):
                return relation
        return None

    def evaluate(self, text: str) -> Tuple[Optional[float], Optional[str], List[Dict[str, Any]]]:
        relation = self._relation(text)
        if not relation:
            return None, None, []

        anchors = self.resolve(text)
        if len(anchors) < 2:
            return None, None, anchors

        first, second = anchors[0], anchors[1]
        a_start, a_end = first["start"], first["end"]
        b_start, b_end = second["start"], second["end"]

        contradiction = False
        if relation == "during":
            contradiction = a_end < b_start or a_start > b_end
        elif relation in {"before", "prior to", "earlier than", "preceding"}:
            contradiction = a_end >= b_start
        elif relation in {"after", "following", "later than", "since"}:
            contradiction = a_start <= b_end

        if contradiction:
            return 0.92, (
                f"Dynamic event-anchor contradiction: '{first['query']}' [{a_start}-{a_end}] "
                f"is incompatible with relation '{relation}' to '{second['query']}' [{b_start}-{b_end}]."
            ), anchors
        return 0.0, (
            f"Dynamic event-anchor relation '{relation}' is temporally compatible: "
            f"'{first['query']}' [{a_start}-{a_end}] vs '{second['query']}' [{b_start}-{b_end}]."
        ), anchors


class Pillar1RetrievalEngine:
    """Pillar 1: Retrieval + NLI Factual Verification + Temporal Consistency Analysis."""

    def __init__(self):
        self.entailment_engine = EvidenceEntailmentEngine()
        self.temporal_engine = TemporalClaimEngine()
        self.event_anchor_resolver = EventTemporalAnchorResolver()
        self.numeric_checker = NumericUnitChecker()
        self.negation_detector = NegationDetector()
        self.causal_checker = CausalDirectionChecker()

    def extract_claims(self, text: str) -> List[str]:
        import time
        t0 = time.perf_counter()
        clean_text = text.strip()
        if not clean_text:
            self.last_claim_extraction_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return []
        # Strip conversational framing prefix if present
        t_clean = re.sub(
            r"^(?:sure|certainly|of course|absolutely|gladly|happy to help|to answer your question)[,!:]?\s*(?:here (?:is|are) (?:the|some|what|your)[^:]*[:.]\s*)?",
            "",
            clean_text,
            flags=re.IGNORECASE,
        ).strip()
        if t_clean:
            clean_text = t_clean
        # Split on sentence boundaries, newlines, semicolons, and clause commas
        raw_claims = re.split(r'(?<=[.!?])\s+|\n+|;\s*|(?<!\d),(?!\d)\s*', clean_text)
        claims = [claim.strip() for claim in raw_claims if len(claim.strip().split()) >= 3]
        if not claims:
            claims = [clean_text]
        self.last_claim_extraction_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return claims

    def _evidence_relevant_to_claim(self, claim: str, item: EvidenceItem) -> bool:
        if has_conflicting_designated_entity(claim, item.snippet, getattr(item, "source_name", "")):
            return False
        evidence_claim = (item.claim or "").strip()
        if not evidence_claim:
            return True
        return (
            evidence_claim.lower() == claim.lower()
            or claim.lower() in evidence_claim.lower()
            or evidence_claim.lower() in claim.lower()
        )

    @staticmethod
    def _score_result(claim: str, item: EvidenceItem, result: dict, idx: int) -> Tuple[float, float, float, str]:
        entailment = result["entailment"]
        contradiction = result["contradiction"]
        neutral = result["neutral"]
        sim_score = float(getattr(item, "score", getattr(item, "similarity_score", 0.88)))

        # Designated entity alignment guard: If a snippet describes a conflicting designated sibling entity
        # (e.g. Apollo 12 vs Apollo 11, Boeing 777 vs Boeing 737, World War II vs World War I),
        # it must not trigger false contradictions against facts asserted about the claim's entity.
        if has_conflicting_designated_entity(claim, item.snippet, getattr(item, "source_name", "")):
            contradiction = min(contradiction, 0.05)
            entailment = min(entailment, 0.05)
            neutral = max(neutral, 0.95)
            return entailment, contradiction, neutral, "neural_deberta_nli"

        # Check for chemical formula contradiction or match
        claim_form = extract_chemical_formula(claim)
        snippet_form = extract_chemical_formula(item.snippet)

        if claim_form and snippet_form:
            c_subj, c_formula = claim_form
            s_subj, s_formula = snippet_form
            if c_subj in s_subj or s_subj in c_subj:
                if c_formula != s_formula:
                    return 0.0, 0.98, 0.02, "deterministic_chemical_formula_rule"
                else:
                    return 0.98, 0.0, 0.02, "deterministic_chemical_formula_rule"

        # Check for chemical composition match (e.g. Water is composed of hydrogen and oxygen)
        claim_comp = extract_chemical_composition(claim)
        if claim_comp:
            c_subj, c_elems = claim_comp
            expected_elems = ELEMENT_MAP.get(c_subj, set())
            snip_lower = item.snippet.lower()
            if c_subj in snip_lower or (c_subj == "water" and "h2o" in snip_lower):
                if c_elems == expected_elems or c_elems.issubset(expected_elems):
                    return 0.98, 0.0, 0.02, "deterministic_chemical_composition_rule"
                else:
                    return 0.0, 0.98, 0.02, "deterministic_chemical_composition_rule"

        # Check for capital city contradiction or entailment using robust structural extractor
        claim_cap = extract_capital_relation(claim)
        snippet_cap = extract_capital_relation(item.snippet)
        if claim_cap and snippet_cap:
            c_city, c_entity = claim_cap
            s_city, s_entity = snippet_cap
            # Check if entities refer to the same target (substring or exact match)
            if c_entity in s_entity or s_entity in c_entity:
                if c_city != s_city and c_city not in s_city and s_city not in c_city:
                    # Clear factual contradiction: evidence names a different capital for the same entity
                    return 0.0, 0.98, 0.02, "deterministic_capital_relation_rule"
                else:
                    # Clear factual entailment: evidence names the same capital for the same entity
                    return 0.98, 0.0, 0.02, "deterministic_capital_relation_rule"

        # SRO extraction & domain verification for scientific constants, counts, and astronomy
        sro = ScientificClaimReasoner.extract_sro(claim)

        # 1. Physical constants and measurement values (speed of light, molar mass, temperature)
        if sro.domain in ("physics_constant", "chemistry", "thermodynamics"):
            snip_l = item.snippet.lower()
            if sro.domain == "physics_constant" and sro.relation == "speed_constant" and sro.numeric_value is not None:
                # Match speed of light in vacuum evidence
                if "speed of light" in snip_l or "light travels" in snip_l:
                    all_speeds = list(re.finditer(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(m/s|km/s|m⋅s−1|meters? per second)", item.snippet, re.IGNORECASE))
                    found_eq = False
                    for m_snip_speed in all_speeds:
                        s_val = float(m_snip_speed.group(1).replace(",", ""))
                        s_unit = m_snip_speed.group(2).lower()
                        is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                            sro.numeric_value, sro.unit or "m/s", s_val, s_unit, "speed", claim
                        )
                        if is_eq:
                            found_eq = True
                            break
                    if found_eq:
                        return 0.98, 0.0, 0.02, "deterministic_scientific_constant_rule"
                    elif "speed of light" in snip_l:
                        c_speed_match = re.search(r"\bspeed\s+of\s+light\s+(?:in\s+vacuum\s+)?(?:is|equals|=)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(m/s|km/s|m⋅s−1|meters? per second)", snip_l)
                        if c_speed_match:
                            s_val = float(c_speed_match.group(1).replace(",", ""))
                            s_unit = c_speed_match.group(2).lower()
                            is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                                sro.numeric_value, sro.unit or "m/s", s_val, s_unit, "speed", claim
                            )
                            if not is_eq:
                                return 0.0, 0.98, 0.02, "deterministic_scientific_constant_rule"

            elif sro.relation == "molar_mass" and sro.numeric_value is not None:
                # Match molar mass in snippet
                if "molar mass" in snip_l or "molecular mass" in snip_l or "g/mol" in snip_l or "18.015" in snip_l:
                    all_mm = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(g/mol|g⋅mol−1|kg/mol|daltons?|grams?\s+per\s+mole|grams)", item.snippet, re.IGNORECASE))
                    found_eq = False
                    for m_snip_mm in all_mm:
                        s_val = float(m_snip_mm.group(1))
                        s_unit = "g/mol"
                        is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                            sro.numeric_value, sro.unit or "g/mol", s_val, s_unit, "molar_mass", claim
                        )
                        if is_eq:
                            found_eq = True
                            break
                    if found_eq:
                        return 0.98, 0.0, 0.02, "deterministic_scientific_constant_rule"
                    elif sro.subject in snip_l:
                        subj_mm_match = re.search(rf"\bmolar\s+mass\s+of\s+{re.escape(sro.subject)}\s+(?:is|=)\s+(\d+(?:\.\d+)?)\b", snip_l)
                        if subj_mm_match:
                            s_val = float(subj_mm_match.group(1))
                            is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                                sro.numeric_value, sro.unit or "g/mol", s_val, "g/mol", "molar_mass", claim
                            )
                            if not is_eq:
                                return 0.0, 0.98, 0.02, "deterministic_scientific_constant_rule"

            elif sro.domain == "thermodynamics" and sro.numeric_value is not None:
                snip_boil = any(w in snip_l for w in ("boil", "boiling"))
                snip_freez = any(w in snip_l for w in ("freez", "melting point", "ice"))
                claim_boil = sro.relation == "boiling_point"
                claim_freez = sro.relation == "freezing_point"
                if claim_boil and snip_boil:
                    m_boil = re.search(r"\bboil\w*\s+(?:at|=)?\s*(\d+(?:\.\d+)?)\s*(°c|°f|k|deg\s*c|degrees?\s+celsius)", snip_l)
                    if m_boil:
                        s_val = float(m_boil.group(1))
                        s_unit = m_boil.group(2).lower()
                        is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                            sro.numeric_value, sro.unit or "°c", s_val, s_unit, "temperature", claim
                        )
                        if is_eq:
                            return 0.98, 0.0, 0.02, "deterministic_scientific_constant_rule"
                        else:
                            return 0.0, 0.98, 0.02, "deterministic_scientific_constant_rule"
                elif claim_freez and snip_freez:
                    m_freez = re.search(r"\bfreez\w*\s+(?:at|=)?\s*(\d+(?:\.\d+)?)\s*(°c|°f|k|deg\s*c|degrees?\s+celsius)", snip_l)
                    if m_freez:
                        s_val = float(m_freez.group(1))
                        s_unit = m_freez.group(2).lower()
                        is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
                            sro.numeric_value, sro.unit or "°c", s_val, s_unit, "temperature", claim
                        )
                        if is_eq:
                            return 0.98, 0.0, 0.02, "deterministic_scientific_constant_rule"
                        else:
                            return 0.0, 0.98, 0.02, "deterministic_scientific_constant_rule"

        # 2. Astronomy: orbits and central body assertions
        if sro.domain == "astronomy":
            snip_l = item.snippet.lower()
            if sro.relation == "centre_of":
                # e.g., "The Earth is the centre of the Solar System"
                if "solar system" in snip_l or "universe" in snip_l:
                    # Check if evidence establishes Sun as centre / heliocentrism
                    if re.search(r"\b(sun|star)\s+(?:is|at)\s+(?:the\s+)?(?:star\s+at\s+the\s+)?(?:centre|center)\b", snip_l) or "heliocentr" in snip_l or "copernic" in snip_l or "planets revolve around the sun" in snip_l or "orbit the sun" in snip_l:
                        if sro.subject in ("earth", "moon", "mars", "jupiter"):
                            return 0.0, 0.98, 0.02, "deterministic_astronomy_rule"
                        elif sro.subject in ("sun", "star"):
                            return 0.98, 0.0, 0.02, "deterministic_astronomy_rule"
            elif sro.relation == "orbits":
                if f"{sro.subject} orbits the {sro.obj}" in snip_l or f"{sro.subject} orbits {sro.obj}" in snip_l or f"orbit around the {sro.obj}" in snip_l or (sro.subject in snip_l and f"orbit the {sro.obj}" in snip_l) or (sro.subject in snip_l and sro.obj in snip_l and ("orbit" in snip_l or "revolve" in snip_l)):
                    return 0.98, 0.0, 0.02, "deterministic_astronomy_rule"

        # Formula / alias match for short equivalence statements "X is Y"
        match = re.match(r'^([A-Za-z0-9\s]+)\s+is\s+([A-Za-z0-9]+)\.?$', claim.strip(), re.IGNORECASE)
        if match:
            subj, obj = match.group(1).strip().lower(), match.group(2).strip().lower()
            snippet_lower = item.snippet.lower()
            if f"{subj} ({obj})" in snippet_lower or f"{subj} ({obj}" in snippet_lower or f"formula {obj}" in snippet_lower or f"formula is {obj}" in snippet_lower:
                return 0.98, 0.0, 0.02, "deterministic_chemical_formula_rule"

        # Aircraft specification & discrete quantity consistency
        num_word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twin": 2, "triple": 3, "quad": 4,
        }
        if sro.domain == "aviation" and sro.relation == "engine_count" and sro.numeric_value is not None:
            c_val = int(round(sro.numeric_value))
            snip_engine_match = re.search(
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twin|triple|quad|\d+)(?:[- ]engine(?:d|s)?|(?:\s+[a-zA-Z0-9&−-]+){0,8}\s+engines?|(?:\s+[a-zA-Z0-9&−-]+){0,8}\s+turbofan engines?)\b",
                item.snippet,
                re.IGNORECASE,
            )
            if snip_engine_match:
                s_raw_num = snip_engine_match.group(1).lower()
                s_val = num_word_map.get(s_raw_num, int(s_raw_num) if s_raw_num.isdigit() else None)
                if s_val is not None:
                    if s_val != c_val:
                        return 0.0, 0.98, 0.02, "deterministic_aircraft_spec_rule"
                    else:
                        return 0.98, 0.0, 0.02, "deterministic_aircraft_spec_rule"

        STOP_NOUNS = {"main", "other", "more", "different", "various", "total", "year", "month", "day", "has", "had", "have", "was", "were", "is", "are"}
        claim_num_matches = re.finditer(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twin|triple|quad|\d+)\s+([a-zA-Z]{3,})\b",
            claim,
            re.IGNORECASE,
        )
        for m in claim_num_matches:
            c_raw_num = m.group(1).lower()
            c_noun = m.group(2).lower().rstrip("s")
            c_val = num_word_map.get(c_raw_num, int(c_raw_num) if c_raw_num.isdigit() else None)
            if c_val is not None and c_noun not in STOP_NOUNS:
                snip_engine_match = re.search(
                    rf"\b(one|two|three|four|five|six|seven|eight|nine|ten|twin|triple|quad|\d+)(?:[- ]engine(?:d|s)?|(?:\s+[a-zA-Z0-9&−-]+){{0,8}}\s+{re.escape(c_noun)}s?)\b",
                    item.snippet,
                    re.IGNORECASE,
                )
                if snip_engine_match:
                    s_raw_num = snip_engine_match.group(1).lower()
                    s_val = num_word_map.get(s_raw_num, int(s_raw_num) if s_raw_num.isdigit() else None)
                    if s_val is not None:
                        if s_val != c_val:
                            return 0.0, 0.98, 0.02, "deterministic_quantity_spec_rule"
                        else:
                            return 0.98, 0.0, 0.02, "deterministic_quantity_spec_rule"

        return entailment, contradiction, neutral, "neural_deberta_nli"

    def evaluate_claims_against_evidence(self, claims: List[str], external_evidence: List[EvidenceItem]) -> Tuple[float, List[EvidenceItem]]:
        import time
        t0 = time.perf_counter()
        if not claims:
            self.last_claim_statuses = []
            self.last_nli_ms = 0.0
            return 0.0, external_evidence
        if not external_evidence:
            self.last_claim_statuses = ["INSUFFICIENT_EVIDENCE" for _ in claims]
            self.last_nli_ms = 0.0
            return 0.5, external_evidence

        claim_error_scores: List[float] = []
        claim_statuses: List[str] = []

        for claim_idx, claim in enumerate(claims):
            sro = ScientificClaimReasoner.extract_sro(claim)

            # 1. Relevance Assessment before NLI aggregation (Phase 16B)
            # Evaluate each evidence item and populate provenance metadata
            candidate_evidence: List[EvidenceItem] = []
            for raw_item in external_evidence:
                rel_score, rel_reason = ScientificClaimReasoner.evaluate_relevance(
                    sro, raw_item.snippet, getattr(raw_item, "source_name", "")
                )
                raw_item.claim = claim
                raw_item.claim_subject = sro.subject
                raw_item.claim_relation = sro.relation
                raw_item.claim_object = sro.obj
                raw_item.relevance_score = round(rel_score, 4)
                if not getattr(raw_item, "retrieval_query", None):
                    raw_item.retrieval_query = sro.raw_claim

                item = raw_item.model_copy() if hasattr(raw_item, "model_copy") else raw_item.copy()
                item.claim = claim
                item.claim_subject = sro.subject
                item.claim_relation = sro.relation
                item.claim_object = sro.obj
                item.relevance_score = round(rel_score, 4)
                if not getattr(item, "retrieval_query", None):
                    item.retrieval_query = sro.raw_claim

                # Filter strictly by relevance threshold >= 0.40
                if rel_score >= 0.40:
                    candidate_evidence.append(item)
                else:
                    # Irrelevant evidence excluded from decision
                    raw_item.entailment_prob = 0.0
                    raw_item.contradiction_prob = 0.0
                    raw_item.neutral_prob = 1.0

            # Condition F: No relevant evidence available for this claim
            if not candidate_evidence:
                claim_statuses.append("INSUFFICIENT_EVIDENCE")
                claim_error_scores.append(0.50)
                continue

            # Sort candidate evidence by relevance score descending
            candidate_evidence.sort(key=lambda x: getattr(x, "relevance_score", 0.0), reverse=True)
            top_candidates = candidate_evidence[:8]

            # 2. Form NLI pairs for relevant candidate evidence
            pairs = [(claim, item.snippet) for item in top_candidates]
            results = self.entailment_engine.classify_batch(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                batch_size=1,
            ) if pairs else []

            # 3. Score results and populate passage-level NLI probabilities
            for idx, (result, item) in enumerate(zip(results, top_candidates)):
                scored = self._score_result(claim, item, result, idx)
                if len(scored) == 4:
                    ent, cont, neut, eval_src = scored
                else:
                    ent, cont, neut = scored
                    eval_src = "neural_deberta_nli"
                item.entailment_prob = round(ent, 4)
                item.contradiction_prob = round(cont, 4)
                item.neutral_prob = round(neut, 4)
                item.evaluation_source = eval_src

                # Mirror scored metrics back to corresponding item in external_evidence
                for raw_item in external_evidence:
                    if raw_item.snippet == item.snippet:
                        raw_item.entailment_prob = item.entailment_prob
                        raw_item.contradiction_prob = item.contradiction_prob
                        raw_item.neutral_prob = item.neutral_prob
                        raw_item.evaluation_source = item.evaluation_source
                        raw_item.claim_subject = item.claim_subject
                        raw_item.claim_relation = item.claim_relation
                        raw_item.claim_object = item.claim_object
                        raw_item.relevance_score = item.relevance_score
                        break

            # 4. Multi-passage Evidence Aggregation (Phase 16D Conditions A-F)
            supporting_passages = [
                it for it in top_candidates
                if it.entailment_prob is not None and it.entailment_prob >= 0.65 and it.entailment_prob > (it.contradiction_prob or 0.0)
            ]
            contradicting_passages = [
                it for it in top_candidates
                if it.contradiction_prob is not None and it.contradiction_prob >= 0.50 and it.contradiction_prob > (it.entailment_prob or 0.0)
            ]
            neutral_passages = [
                it for it in top_candidates
                if it not in supporting_passages and it not in contradicting_passages
            ]

            if supporting_passages and contradicting_passages:
                max_supp_ent = max(it.entailment_prob for it in supporting_passages)
                max_cont_prob = max(it.contradiction_prob for it in contradicting_passages)
                if max_supp_ent >= 0.70 and max_cont_prob >= 0.70:
                    # Genuine conflicting relevant evidence produces an uncertainty state
                    claim_statuses.append("INSUFFICIENT_EVIDENCE")
                    claim_error_scores.append(0.50)
                elif max_supp_ent >= 0.80 and max_cont_prob < 0.70:
                    # Strongly entailed authoritative evidence dominates weaker counter-signals
                    claim_statuses.append("VERIFIED")
                    claim_error_scores.append(max(0.0, round(1.0 - max_supp_ent, 4)))
                elif max_cont_prob >= 0.80 and max_supp_ent < 0.70:
                    # Strongly contradictory evidence dominates weaker signals
                    claim_statuses.append("CONTRADICTED")
                    claim_error_scores.append(max_cont_prob)
                else:
                    claim_statuses.append("INSUFFICIENT_EVIDENCE")
                    claim_error_scores.append(0.50)
            elif contradicting_passages and not supporting_passages:
                # Relevant contradictory evidence determines verdict (Condition B, Condition D)
                max_cont = max(it.contradiction_prob for it in contradicting_passages)
                claim_statuses.append("CONTRADICTED")
                claim_error_scores.append(max_cont)
            elif supporting_passages and not contradicting_passages:
                # Relevant supporting evidence determines verdict (Condition A, Condition C)
                max_ent = max(it.entailment_prob for it in supporting_passages)
                claim_statuses.append("VERIFIED")
                claim_error_scores.append(max(0.0, round(1.0 - max_ent, 4)))
            else:
                # Relevant but neutral evidence does NOT become contradiction (Condition E)
                claim_statuses.append("INSUFFICIENT_EVIDENCE")
                claim_error_scores.append(0.50)

        self.last_claim_statuses = claim_statuses
        self.last_nli_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.last_nli_batch_metrics = getattr(self.entailment_engine, "last_batch_metrics", {})
        factual_error = sum(claim_error_scores) / len(claim_error_scores) if claim_error_scores else 0.50
        return round(factual_error, 4), external_evidence

    def analyze(self, text: str, provided_evidence: List[EvidenceItem] = None, query: Optional[str] = None) -> Pillar1Result:
        if provided_evidence is None:
            provided_evidence = []
        claims = self.extract_claims(text)
        fe_score, evidence = self.evaluate_claims_against_evidence(claims, provided_evidence)

        temp_res = self.temporal_engine.analyze_claim(text, query=query, evidence_items=provided_evidence)
        event_anchor_score = None
        event_anchor_reasoning = None
        event_anchors: List[Dict[str, Any]] = []

        # Phase 6: only invoke dynamic event anchoring when explicit years are
        # absent and the temporal engine identifies relational language.
        if temp_res.temporal_status == TemporalStatus.TIME_RELATIVE and not temp_res.extracted_years:
            event_anchor_score, event_anchor_reasoning, event_anchors = self.event_anchor_resolver.evaluate(text)

        if temp_res.temporal_inconsistency_score > 0.0:
            fe_score = round(max(fe_score, temp_res.temporal_inconsistency_score), 4)
        elif event_anchor_score is not None and event_anchor_score > 0.0:
            fe_score = round(max(fe_score, event_anchor_score), 4)
        elif temp_res.protected_from_temporal_penalty and temp_res.modality != EpistemicModality.ASSERTED_FACT:
            if temp_res.modality in {
                EpistemicModality.PREDICTION,
                EpistemicModality.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL,
                EpistemicModality.CONDITIONAL,
                EpistemicModality.FICTIONAL,
            }:
                fe_score = 0.0

        # Symbolic checks against top retrieved evidence (relevant evidence only)
        # Only evaluate numeric/negation/causal checks on single atomic assertions (len(claims) <= 1)
        # to avoid false conflicts between distinct numbers in multi-claim texts (e.g. 0°C freezing vs 100°C boiling)
        if len(claims) <= 1:
            rel_evidence = [
                e for e in evidence
                if getattr(e, "relevance_score", None) is not None
                and getattr(e, "relevance_score", 0.0) >= 0.40
                and getattr(e, "snippet", "")
            ]
            ev_snippets = [e.snippet for e in rel_evidence]
            if ev_snippets:
                ev_combined = " ".join(ev_snippets[:3])
                num_status, num_penalty, _ = self.numeric_checker.check_consistency(text, ev_combined)
                if num_status in (NumericUnitStatus.NUMERIC_CONFLICT, NumericUnitStatus.SCALE_CONFLICT, NumericUnitStatus.UNIT_CONFLICT):
                    fe_score = round(max(fe_score, num_penalty), 4)

                pol_res = self.negation_detector.analyze(text, ev_combined)
                if pol_res.negation_inversion_detected or pol_res.antonym_inversion_detected:
                    fe_score = round(max(fe_score, pol_res.confidence_penalty), 4)

                for snip in ev_snippets[:3]:
                    caus_res = self.causal_checker.check_inversion(text, snip)
                    if caus_res.is_inversion_detected:
                        fe_score = round(max(fe_score, caus_res.confidence_penalty), 4)
                        break

        # Check deterministic symbolic arithmetic & unit claims
        sym_reasoning = None
        arith_res = evaluate_arithmetic_claim(text)
        if not arith_res and query:
            arith_res = evaluate_arithmetic_claim(f"{query} = {text}")
        if arith_res and arith_res.get("verified"):
            if arith_res.get("is_consistent") is True:
                fe_score = 0.0
                sym_reasoning = f"Symbolic Arithmetic Verified: {arith_res.get('explanation')}"
                self.last_claim_statuses = ["VERIFIED" for _ in claims]
            elif arith_res.get("is_consistent") is False:
                fe_score = 1.0
                sym_reasoning = f"Symbolic Arithmetic Contradiction: {arith_res.get('explanation')}"
                self.last_claim_statuses = ["CONTRADICTED" for _ in claims]

        unit_res = evaluate_unit_claim(text)
        if unit_res and unit_res.get("verified"):
            if unit_res.get("is_consistent") is True:
                fe_score = 0.0
                sym_reasoning = f"Symbolic Unit Verified: {unit_res.get('explanation')}"
                self.last_claim_statuses = ["VERIFIED" for _ in claims]
            elif unit_res.get("is_consistent") is False:
                fe_score = 1.0
                sym_reasoning = f"Symbolic Unit Contradiction: {unit_res.get('explanation')}"
                self.last_claim_statuses = ["CONTRADICTED" for _ in claims]

        if not claims:
            reasoning = "No discrete factual claims identified."
        elif sym_reasoning:
            reasoning = sym_reasoning
        elif temp_res.temporal_inconsistency_score > 0.0:
            reasoning = f"Temporal Inconsistency: {temp_res.reasoning}"
        elif event_anchor_score is not None and event_anchor_score > 0.0:
            reasoning = f"Temporal Event-Anchor Inconsistency: {event_anchor_reasoning}"
        elif event_anchor_reasoning:
            reasoning = f"Temporal Event-Anchor Check: {event_anchor_reasoning}"
        elif temp_res.protected_from_temporal_penalty and temp_res.modality != EpistemicModality.ASSERTED_FACT:
            reasoning = f"Protected Epistemic Modality ({temp_res.modality.value}): temporal penalty suppressed."
        elif not provided_evidence:
            reasoning = f"Identified {len(claims)} factual claim(s), but no external evidence was available. Factual status remains uncertain."
        elif fe_score < 0.20:
            reasoning = f"High factual grounding. {len(claims)} claim(s) are strongly entailed by retrieved evidence."
        elif fe_score < 0.50:
            reasoning = f"Moderate factual grounding. {len(claims)} claim(s) have partial evidence support."
        elif fe_score < 0.70:
            reasoning = f"Insufficient or conflicting evidence detected for {len(claims)} claim(s)."
        else:
            reasoning = f"Strong factual inconsistency detected. Retrieved evidence contradicts one or more of the {len(claims)} analyzed claim(s)."

        retrieved_passages = [item.snippet for item in evidence if item.snippet]
        bm25_scores, dense_scores = [], []
        for claim in claims:
            for item in evidence:
                w_claim = set(re.findall(r'\w+', claim.lower()))
                w_snippet = set(re.findall(r'\w+', item.snippet.lower()))
                bm25_scores.append(len(w_claim.intersection(w_snippet)) / (len(w_claim) + 1e-6))
                dense_scores.append(item.similarity_score)

        avg_bm25 = round(sum(bm25_scores) / len(bm25_scores), 4) if bm25_scores else 0.0
        avg_dense = round(sum(dense_scores) / len(dense_scores), 4) if dense_scores else 0.0
        cross_encoder_avg = round(0.7 * avg_dense + 0.3 * avg_bm25, 4)
        citation_conf = round(max(0.0, min(1.0, 1.0 - fe_score * 0.5 + avg_dense * 0.5)), 4)

        # Diagnostic metadata is attached non-invasively for Phase 6 reports.
        self.last_event_anchor_diagnostics = {
            "anchors": event_anchors,
            "score": event_anchor_score,
            "reasoning": event_anchor_reasoning,
            "lookups": self.event_anchor_resolver.last_lookup_count,
        }

        return Pillar1Result(
            claims=claims,
            evidence=evidence,
            factual_error_score=fe_score,
            reasoning=reasoning,
            retrieved_passages=retrieved_passages,
            citation_confidence_score=citation_conf,
            dense_retrieval_score=avg_dense,
            bm25_retrieval_score=avg_bm25,
            cross_encoder_score=cross_encoder_avg,
        )
