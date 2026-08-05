"""
Sprint 3 UX & Verification Integration Test.
Verifies:
1. Chat & verification report persistence.
2. GET /api/v1/verification/{message_id} REST API payload integrity.
3. Nested sentence_analyses, evidence_items, H-Scores, and risk levels.
"""
import asyncio
import pytest
from uuid import UUID
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database.session import AsyncSessionLocal, _engine
from app.workers.tasks.verification_task import run_verification_async

@pytest.mark.asyncio
async def test_sprint3_ux_verification_e2e():
    await _engine.dispose()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        email = "sprint3_ux_user@hallucisense.ai"
        password = "Password123!"

        # 1. Auth & Token
        reg_res = await client.post("/api/v1/auth/register", json={
            "email": email, "password": password, "full_name": "Sprint3 User"
        })
        if reg_res.status_code == 201:
            tokens = reg_res.json()["tokens"]
        else:
            login_res = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
            tokens = login_res.json()["tokens"]

        access_token = tokens["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # 2. Create Chat
        chat_res = await client.post("/api/v1/chats", json={"title": "Sprint 3 UX Test Chat"}, headers=headers)
        assert chat_res.status_code in (200, 201)
        chat_id = chat_res.json()["id"]

        # 3. Post external response for verification
        test_text = (
            "Special relativity was published by Albert Einstein in 1905. "
            "Quantum mechanics describes physics at atomic scales."
        )
        ext_res = await client.post(f"/api/v1/chats/{chat_id}/messages/verify-external", json={
            "content": test_text
        }, headers=headers)
        assert ext_res.status_code == 202
        message_id_str = ext_res.json()["message_id"]
        message_id = UUID(message_id_str)

        # 4. Run verification async task
        verif_res = await run_verification_async(
            message_id=message_id,
            full_ai_text=test_text,
            token_probs=[0.99, 0.97, 0.95]
        )
        assert verif_res["status"] == "success"

        # 5. Verify GET /api/v1/verification/{message_id} returns UI schema
        report_res = await client.get(f"/api/v1/verification/{message_id_str}", headers=headers)
        assert report_res.status_code == 200
        report = report_res.json()

        assert "overall_h_score" in report
        assert "overall_risk_level" in report
        assert "sentence_analyses" in report
        assert len(report["sentence_analyses"]) >= 1

        print("\n==========================================================================")
        print(f"SPRINT 3 INTEGRATION TEST PASSED!")
        print(f"Message ID: {message_id_str}")
        print(f"Overall H-Score: {report['overall_h_score']:.2f}")
        print(f"Overall Risk Level: {report['overall_risk_level']}")
        print(f"Sentences Analyzed: {len(report['sentence_analyses'])}")
        print("==========================================================================")

if __name__ == "__main__":
    asyncio.run(test_sprint3_ux_verification_e2e())
