"""
Async Integration Test Script for HalluciSense WebSocket Chat Streaming Pipeline.
"""
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from app.main import app
from app.database.session import _engine

@pytest.mark.asyncio
async def test_async_websocket_streaming():
    # Dispose engine pool attached to old loop
    await _engine.dispose()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        email = "async_ws_user@hallucisense.ai"
        password = "Password123!"

        print("\n[ASYNC TEST] Registering user...")
        reg_res = await client.post("/api/v1/auth/register", json={
            "email": email,
            "password": password,
            "full_name": "Async WS User"
        })
        
        if reg_res.status_code == 201:
            tokens = reg_res.json()["tokens"]
        else:
            login_res = await client.post("/api/v1/auth/login", json={
                "email": email,
                "password": password
            })
            assert login_res.status_code == 200, f"Login failed: {login_res.text}"
            tokens = login_res.json()["tokens"]

        access_token = tokens["access_token"]
        assert access_token, "No access token returned"
        print(f"[ASYNC TEST] Auth SUCCESS! Access Token len: {len(access_token)}")

        print("[ASYNC TEST] Creating chat...")
        headers = {"Authorization": f"Bearer {access_token}"}
        chat_res = await client.post("/api/v1/chats", json={
            "title": "Async WS Chat",
            "model_used": "gemini-3.1-pro"
        }, headers=headers)
        assert chat_res.status_code in (200, 201), f"Create chat failed: {chat_res.text}"
        chat_id = chat_res.json()["id"]
        print(f"[ASYNC TEST] Chat created! ID: {chat_id}")

        print("[ASYNC TEST] Testing WebSocket streaming via TestClient...")
        test_client = TestClient(app)
        ws_url = f"/api/v1/chats/{chat_id}/messages/stream?token={access_token}"

        tokens_received = []
        verification_dispatched = False

        with test_client.websocket_connect(ws_url) as ws:
            payload = {
                "chat_id": chat_id,
                "content": "Explain what speed of light is in one sentence.",
                "model": "gemini-3.1-pro"
            }
            ws.send_json(payload)
            print("[ASYNC TEST] Sent prompt payload. Receiving tokens...\n")

            while True:
                try:
                    data = ws.receive_json()
                    msg_type = data.get("type")
                    if msg_type == "token":
                        text = data.get("text", "")
                        tokens_received.append(text)
                        print(text, end="", flush=True)
                    elif msg_type == "verification_dispatched":
                        print(f"\n\n[ASYNC TEST] Verification Dispatched! Msg ID: {data.get('message_id')}")
                        verification_dispatched = True
                        break
                    elif msg_type == "error":
                        raise RuntimeError(f"Error frame: {data.get('error')}")
                except Exception as e:
                    print(f"\n[ASYNC TEST] Stream finished or loop closed: {e}")
                    break

        print(f"\n[ASYNC TEST] Total tokens received: {len(tokens_received)}")
        assert len(tokens_received) > 0, "No tokens received!"
        assert verification_dispatched, "Verification dispatched missing!"

        print("[ASYNC TEST] Checking DB history...")
        hist_res = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
        assert hist_res.status_code == 200
        messages = hist_res.json().get("items", [])
        print(f"[ASYNC TEST] Saved messages in DB: {len(messages)}")
        assert len(messages) >= 2

        print("\n=====================================================================")
        print("SUCCESS: HalluciSense WebSocket Chat Streaming Pipeline 100% Verified!")
        print("=====================================================================")

if __name__ == "__main__":
    asyncio.run(test_async_websocket_streaming())
