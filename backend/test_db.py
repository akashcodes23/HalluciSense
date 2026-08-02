import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import AsyncSessionLocal
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.models.sentence_analysis import SentenceAnalysis

async def main():
    async with AsyncSessionLocal() as session:
        # Try different option syntaxes
        try:
            stmt = select(Message).options(
                selectinload(Message.verification_report).selectinload(VerificationReport.sentence_analyses)
            )
            print("Syntax 1 worked!")
        except Exception as e:
            print("Syntax 1 failed:", type(e).__name__, str(e))
            
        try:
            stmt = select(Message).options(
                selectinload(Message.verification_report).options(selectinload(VerificationReport.sentence_analyses))
            )
            print("Syntax 2 worked!")
        except Exception as e:
            print("Syntax 2 failed:", type(e).__name__, str(e))

asyncio.run(main())
