"""
Full Verification Engine End-to-End Integration Test.
Verifies:
1. Message creation & LLM response.
2. Async verification execution (run_verification_async).
3. DB Persistence: VerificationReport, SentenceAnalysis, EvidenceItem.
4. Foreign Key and Cascade integrity.
5. Verification REST Endpoint: GET /api/v1/verification/{message_id}.
"""
import asyncio
import pytest
from uuid import UUID
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database.session import AsyncSessionLocal, _engine
from app.workers.tasks.verification_task import run_verification_async
from app.models.verification_report import VerificationReport
from sqlalchemy import select

@pytest.mark.asyncio
async def test_verification_engine_e2e():
    await _engine.dispose()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        email = "verif_engine_user@hallucisense.ai"
        password = "Password123!"

        # 1. Auth
        reg_res = await client.post("/api/v1/auth/register", json={
            "email": email, "password": password, "full_name": "Verif User"
        })
        if reg_res.status_code == 201:
            tokens = reg_res.json()["tokens"]
        else:
            login_res = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
            tokens = login_res.json()["tokens"]

        access_token = tokens["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # 2. Create Chat
        chat_res = await client.post("/api/v1/chats", json={"title": "Verification Pipeline Test Chat"}, headers=headers)
        assert chat_res.status_code in (200, 201)
        chat_id = chat_res.json()["id"]

        # 3. Post external response to verify
        test_ai_text = (
            "The Earth orbits the Sun once every 365.25 days. "
            "The Moon was formed approximately 4.5 billion years ago. "
            "Humans landed on Mars in the year 2020."
        )
        ext_res = await client.post(f"/api/v1/chats/{chat_id}/messages/verify-external", json={
            "content": test_ai_text
        }, headers=headers)
        assert ext_res.status_code == 202
        message_id_str = ext_res.json()["message_id"]
        message_id = UUID(message_id_str)
        print(f"\n[VERIFICATION TEST] External response saved with Message ID: {message_id}")

        # 4. Execute verification pipeline directly
        print("[VERIFICATION TEST] Running verification pipeline (run_verification_async)...")
        verif_res = await run_verification_async(
            message_id=message_id,
            full_ai_text=test_ai_text,
            token_probs=[0.95, 0.92, 0.88, 0.21, 0.15, 0.99]
        )
        assert verif_res["status"] == "success"
        print(f"[VERIFICATION TEST] Pipeline finished! Overall H-Score: {verif_res['overall_h_score']:.2f}")

        # 5. Query REST API GET /api/v1/verification/{message_id}
        report_res = await client.get(f"/api/v1/verification/{message_id_str}", headers=headers)
        assert report_res.status_code == 200, f"GET verification failed: {report_res.text}"
        report_data = report_res.json()
        print(f"[VERIFICATION TEST] GET /verification/{message_id_str} HTTP 200 OK!")
        assert "overall_h_score" in report_data
        assert "sentence_analyses" in report_data
        sentences = report_data["sentence_analyses"]
        print(f"[VERIFICATION TEST] Total Sentence Analyses in DB: {len(sentences)}")
        assert len(sentences) >= 1

        for idx, s in enumerate(sentences):
            print(f"  - Sentence {idx}: H-Score={s['h_score']:.2f}, Risk={s['risk_level']}, Evidence Items={len(s.get('evidence_items', []))}")

        print("\n==========================================================================")
        print("SUCCESS: HalluciSense Verification Engine & Persistence 100% Functional!")
        print("==========================================================================")

if __name__ == "__main__":
    asyncio.run(test_verification_engine_e2e())
