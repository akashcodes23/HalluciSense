import asyncio
import websockets
import json
import uuid
from app.core.security import create_access_token

async def test_ws():
    # Create a dummy user ID and token
    user_id = uuid.uuid4()
    token = create_access_token(user_id)
    chat_id = uuid.uuid4()
    
    uri = f"ws://127.0.0.1:8000/api/v1/chats/{chat_id}/messages/stream?token={token}"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket!")
            
            # Send initial message
            msg = {"chat_id": str(chat_id), "content": "What is 2+2?", "model": "gemini-flash-latest"}
            await websocket.send(json.dumps(msg))
            print("Sent request.")
            
            # Receive chunks
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "token":
                    print(f"Received token: {data.get('text')}")
                else:
                    print(f"Received other: {data}")
                    
    except Exception as e:
        print("WebSocket exception:", e)

asyncio.run(test_ws())
