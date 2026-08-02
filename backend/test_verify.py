import asyncio
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.knowledge.retriever import HybridRetriever
from app.core.engine.types import EvidenceItem

async def run_tests():
    pipeline = HallucinationDetectionPipeline(alpha=0.5, beta=0.3, gamma=0.2)
    retriever = HybridRetriever()
    statements = [
        "What is the capital of France?",
        "India has 35 states.",
        "The Sun revolves around Earth.",
        "Barack Obama is the current President of the USA.",
        "Water boils at 100°C at sea level."
    ]
    
    for stmt in statements:
        print(f"--- Analyzing: {stmt} ---")
        try:
            raw_evidence = retriever.retrieve([stmt])
            evidence_items = []
            for e in raw_evidence:
                evidence_items.append(EvidenceItem(
                    claim=stmt,
                    snippet=e["snippet"],
                    source_name=e["source_name"],
                    source_url=e.get("source_url", ""),
                    similarity_score=e.get("similarity_score", 0.9),
                    is_supporting=e.get("is_supporting", True)
                ))
                
            report = pipeline.analyze_response(
                full_text=stmt, 
                token_probabilities=[0.9]*len(stmt.split()),
                evidence_items=evidence_items
            )
            print(f"Overall H-Score: {report.overall_h_score}")
            print(f"Risk Level: {report.overall_risk_level}")
            print(f"P1 (Retrieval Error): {report.pillar1_summary.factual_error_score}")
            print(f"P2 (Confidence Gap): {report.pillar2_summary.confidence_gap_score}")
            print(f"P3 (Consistency Fail): {report.pillar3_summary.consistency_failure_score}")
            print(f"Reasoning: {report.sentence_analyses[0].reasoning if report.sentence_analyses else 'None'}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
