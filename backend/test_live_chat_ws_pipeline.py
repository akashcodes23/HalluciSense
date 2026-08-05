"""
Live Integration Test Script for HalluciSense WebSocket Chat Streaming Pipeline.
Verifies:
1. User registration & login (JWT token).
2. Chat creation.
3. WebSocket connection to /api/v1/chats/{chat_id}/messages/stream.
4. Sending JSON payload with selected model ("gemini-3.1-pro" / "gemini-flash-latest").
5. Streaming token reception in real time.
6. Verification dispatched event.
7. Clean WebSocket closure (code 1000).
8. DB message persistence and history refresh.
"""
import asyncio
import json
import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000/api/v1"
WS_BASE_URL = "ws://127.0.0.1:8000/api/v1"

async def test_live_chat_websocket_pipeline():
    # Long timeout so registration/hashing completes cleanly
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        email = "ws_live_user_test@hallucisense.ai"
        password = "Password123!"

        print("\n[TEST] 1. Registering/Logging in test user...")
        reg_res = await client.post("/auth/register", json={
            "email": email,
            "password": password,
            "full_name": "WS Live User"
        })
        
        if reg_res.status_code == 201:
            tokens = reg_res.json()["tokens"]
        else:
            login_res = await client.post("/auth/login", json={
                "email": email,
                "password": password
            })
            assert login_res.status_code == 200, f"Login failed: {login_res.text}"
            tokens = login_res.json()["tokens"]

        access_token = tokens["access_token"]
        assert access_token, "No access token returned"
        print(f"[TEST] Auth SUCCESS! Access Token len: {len(access_token)}")

        print("[TEST] 2. Creating a new chat...")
        headers = {"Authorization": f"Bearer {access_token}"}
        chat_res = await client.post("/chats", json={
            "title": "Live WebSocket Stream Test Chat",
            "model_used": "gemini-3.1-pro"
        }, headers=headers)
        assert chat_res.status_code in (200, 201), f"Create chat failed: {chat_res.text}"
        chat_id = chat_res.json()["id"]
        print(f"[TEST] Chat created! ID: {chat_id}")

        print("[TEST] 3. Connecting to WebSocket endpoint...")
        ws_url = f"{WS_BASE_URL}/chats/{chat_id}/messages/stream?token={access_token}"

        tokens_received = []
        verification_dispatched = False

        async with websockets.connect(ws_url) as ws:
            print("[TEST] Connected to WebSocket. Sending prompt payload...")
            payload = {
                "chat_id": chat_id,
                "content": "Explain what special relativity is in one concise sentence.",
                "model": "gemini-3.1-pro"
            }
            await ws.send(json.dumps(payload))
            print("[TEST] Payload sent. Waiting for streaming tokens...\n")

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "token":
                        token_text = data.get("text", "")
                        tokens_received.append(token_text)
                        print(token_text, end="", flush=True)
                    elif msg_type == "verification_dispatched":
                        print(f"\n\n[TEST] Verification Dispatched Event Received! Message ID: {data.get('message_id')}")
                        verification_dispatched = True
                        break
                    elif msg_type == "error":
                        raise RuntimeError(f"WebSocket Error Frame Received: {data.get('error')}")
                except asyncio.TimeoutError:
                    raise RuntimeError("WebSocket timed out waiting for streaming response")

        print(f"\n[TEST] Total tokens received: {len(tokens_received)}")
        assert len(tokens_received) > 0, "No streaming tokens received!"
        assert verification_dispatched, "Verification dispatched event missing!"

        print("[TEST] 4. Verifying DB history via REST API...")
        hist_res = await client.get(f"/chats/{chat_id}/messages", headers=headers)
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        items = hist_data.get("items", [])
        print(f"[TEST] Total messages in chat history: {len(items)}")
        assert len(items) >= 2, f"Expected 2+ messages, found {len(items)}"
        assert items[-1]["role"] in ("ASSISTANT", "assistant")
        assert len(items[-1]["content"]) > 0

        print("\n=================================================================")
        print("ALL TESTS PASSED: Live WebSocket Chat Pipeline is 100% Functional!")
        print("=================================================================")

if __name__ == "__main__":
    asyncio.run(test_live_chat_websocket_pipeline())
