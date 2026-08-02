import asyncio
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        msg = await ws.recv()
        if msg == "verification_dispatched":
            print("Client closing")
            await ws.close(1000)
asyncio.run(main())
