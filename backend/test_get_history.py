import asyncio
from uuid import UUID
from app.database.session import AsyncSessionLocal
from app.modules.messages.service import MessageService
from app.models.user import User
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        # Get a user and their chat
        result = await session.execute(select(User).limit(1))
        user = result.scalars().first()
        from app.models.chat import Chat
        chat_res = await session.execute(select(Chat).where(Chat.user_id == user.id).limit(1))
        chat = chat_res.scalars().first()
        
        if chat:
            print("Chat ID:", chat.id)
            service = MessageService(session)
            msgs, total = await service.get_chat_messages(chat.id, user.id)
            print(f"Total messages: {total}")
            for m in msgs:
                print(m.id, m.role, m.content)

asyncio.run(main())
