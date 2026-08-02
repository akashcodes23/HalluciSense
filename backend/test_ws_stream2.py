import asyncio
import websockets
import json
import uuid
from sqlalchemy import select
from app.core.security import create_access_token
from app.database.session import AsyncSessionLocal
from app.models.chat import Chat
from app.models.user import User

async def setup():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == "test_ws@test.com"))
        user = result.scalars().first()
        if not user:
            user = User(email="test_ws@test.com", hashed_password="pw", full_name="Test")
            session.add(user)
            await session.flush()
        chat = Chat(user_id=user.id, title="Test WS")
        session.add(chat)
        await session.commit()
        return user.id, chat.id

async def test_ws():
    user_id, chat_id = await setup()
    token = create_access_token(str(user_id))
    
    uri = f"ws://127.0.0.1:8000/api/v1/chats/{chat_id}/messages/stream?token={token}"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket!")
            
            # Send initial message
            msg = {"chat_id": str(chat_id), "content": "hello", "model": "gemini-flash-latest"}
            await websocket.send(json.dumps(msg))
            print("Sent request.")
            
            # Receive chunks
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "token":
                    print(f"Token: {data.get('text')}")
                elif data.get("type") == "verification_dispatched":
                    print("Verification dispatched!")
                    # simulate frontend closing
                    await websocket.close(1000)
                else:
                    print(f"Received other: {data}")
                    
    except Exception as e:
        print("WebSocket exception:", type(e), e)

asyncio.run(test_ws())
