import asyncio
from app.database.session import AsyncSessionLocal
from sqlalchemy import text

async def clear_processing():
    async with AsyncSessionLocal() as session:
        await session.execute(text("UPDATE messages SET verification_status = 'VERIFIED' WHERE verification_status = 'PROCESSING'"))
        await session.commit()
        print("✅ Cleared processing messages!")

asyncio.run(clear_processing())
