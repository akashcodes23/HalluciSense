import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/api/v1/chats/378c9d3d-4d84-426a-ab3c-c20f83619883/messages/stream?token=dummy"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            await websocket.send(json.dumps({"content": "hello"}))
            response = await websocket.recv()
            print("Received:", response)
    except Exception as e:
        print("Failed to connect:", e)

asyncio.run(test_ws())
