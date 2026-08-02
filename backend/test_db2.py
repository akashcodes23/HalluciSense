import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import AsyncSessionLocal
from app.models.message import Message

async def main():
    async with AsyncSessionLocal() as session:
        try:
            stmt = select(Message).options(
                selectinload(Message.verification_report)
                .selectinload("sentence_analyses")
                .selectinload("evidence_items")
            )
            print("String chaining worked!")
        except Exception as e:
            print("String chaining failed:", type(e).__name__, str(e))

asyncio.run(main())
