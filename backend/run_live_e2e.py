import asyncio
import json
import websockets
import httpx
import time
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database.session import AsyncSessionLocal
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.models.sentence_analysis import SentenceAnalysis
from app.models.evidence_item import EvidenceItem

BASE_URL = "http://127.0.0.1:8000/api/v1"
WS_URL = "ws://127.0.0.1:8000/api/v1"

async def main():
    print("=== STARTING LIVE E2E VALIDATION ===")
    
    # 1. Register / Login user
    async with httpx.AsyncClient() as client:
        # Register a unique user or login
        user_email = f"e2e_user_{int(time.time())}@example.com"
        reg_resp = await client.post(
            f"{BASE_URL}/auth/register",
            json={
                "email": user_email,
                "password": "Password123!",
                "full_name": "E2E Live User"
            }
        )
        assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"
        data = reg_resp.json()
        token = data["tokens"]["access_token"]
        print(f"1. Authentication: PASS (User: {user_email})")

        # 2. Create Chat
        headers = {"Authorization": f"Bearer {token}"}
        chat_resp = await client.post(
            f"{BASE_URL}/chats",
            headers=headers,
            json={"title": "E2E Test Chat", "model_used": "ollama"}
        )
        assert chat_resp.status_code == 201, f"Create chat failed: {chat_resp.text}"
        chat_id = chat_resp.json()["id"]
        print(f"2. Create Chat: PASS (Chat ID: {chat_id})")

    # 3. Connect via WebSocket and send prompt
    ws_endpoint = f"{WS_URL}/chats/{chat_id}/messages/stream?token={token}"
    prompt = "What is the capital of France? Answer in one sentence."
    print(f"3. Sending Chat Request via WebSocket: '{prompt}'")
    
    ai_message_id = None
    ai_text = ""

    async with websockets.connect(ws_endpoint) as websocket:
        await websocket.send(json.dumps({"content": prompt}))
        
        while True:
            try:
                msg_raw = await websocket.recv()
                msg = json.loads(msg_raw)
                if msg.get("type") == "token":
                    ai_text += msg.get("text", "")
                elif msg.get("type") == "verification_dispatched":
                    ai_message_id = msg.get("message_id")
                    print(f"   Received verification_dispatched: AI Message ID = {ai_message_id}")
                    break
            except websockets.exceptions.ConnectionClosed:
                break

    assert ai_message_id is not None, "Failed to capture AI Message ID from WebSocket!"
    print(f"   AI Response Generated: '{ai_text.strip()}'")
    print("4. Real Chat Request & Dispatched: PASS")

    # 4. Wait for Celery worker to finish verification task
    print("5. Waiting for Celery verification task to complete...")
    msg_uuid = UUID(ai_message_id)
    
    max_wait = 120 # seconds
    start = time.time()
    report_db = None

    while time.time() - start < max_wait:
        await asyncio.sleep(2)
        async with AsyncSessionLocal() as session:
            stmt = (
                select(VerificationReport)
                .where(VerificationReport.message_id == msg_uuid)
                .options(
                    selectinload(VerificationReport.sentence_analyses)
                    .selectinload(SentenceAnalysis.evidence_items)
                )
            )
            res = await session.execute(stmt)
            report_db = res.scalars().first()
            if report_db is not None:
                break

    assert report_db is not None, f"Verification report was not persisted in DB within {max_wait}s!"
    print("   VerificationReport found in Database!")

    # 5. Database Verification
    print("6. Database Verification:")
    print(f"   - Report ID: {report_db.id}")
    print(f"   - Message ID: {report_db.message_id}")
    print(f"   - Overall H-Score: {report_db.overall_h_score}")
    print(f"   - Overall Risk Level: {report_db.overall_risk_level}")
    print(f"   - Factual Error Score: {report_db.factual_error_score}")
    print(f"   - Confidence Gap Score: {report_db.confidence_gap_score}")
    print(f"   - Consistency Failure Score: {report_db.consistency_failure_score}")
    print(f"   - Weights Used: {report_db.weights_used}")
    print(f"   - Pillar 1 Summary: {report_db.pillar1_summary}")
    print(f"   - Pillar 2 Summary: {report_db.pillar2_summary}")
    print(f"   - Pillar 3 Summary: {report_db.pillar3_summary}")
    print(f"   - Corrected Response: {report_db.corrected_response}")
    print(f"   - Processing Time (ms): {report_db.processing_time_ms}")
    print(f"   - Sentence Analyses Count: {len(report_db.sentence_analyses)}")
    
    for idx, sa in enumerate(report_db.sentence_analyses):
        print(f"     Sentence #{idx}: '{sa.sentence_text}' [range {sa.start_char}-{sa.end_char}]")
        print(f"       h_score={sa.h_score}, risk={sa.risk_level}, color={sa.color_code}")
        print(f"       FE={sa.factual_error}, CG={sa.confidence_gap}, CF={sa.consistency_failure}")
        print(f"       reasoning: {sa.reasoning}")
        print(f"       evidence items count: {len(sa.evidence_items)}")

    # 6. API Verification (GET /chats/{chat_id}/messages)
    print("7. API Round-Trip Verification:")
    async with httpx.AsyncClient() as client:
        msg_resp = await client.get(f"{BASE_URL}/chats/{chat_id}/messages", headers=headers)
        assert msg_resp.status_code == 200, f"GET messages failed: {msg_resp.text}"
        msgs = msg_resp.json()["items"]
        ai_msg_api = next((m for m in msgs if m["id"] == ai_message_id), None)
        assert ai_msg_api is not None, "AI Message not found in API response!"
        assert ai_msg_api["verification_report"] is not None, "verification_report missing in API response!"
        api_rep = ai_msg_api["verification_report"]
        print(f"   - API Report overall_h_score: {api_rep['overall_h_score']}")
        print(f"   - API Report overall_risk_level: {api_rep['overall_risk_level']}")
        print(f"   - API Sentence Analyses: {len(api_rep['sentence_analyses'])} items")

    print("=== LIVE E2E VALIDATION SUCCESSFUL ===")

if __name__ == "__main__":
    asyncio.run(main())
