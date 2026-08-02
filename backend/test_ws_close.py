import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK

async def main():
    async with websockets.serve(server, "127.0.0.1", 8765):
        await asyncio.Future()  # run forever

async def server(websocket):
    print("Client connected")
    await websocket.send("verification_dispatched")
    await asyncio.sleep(0.1) # Simulate client closing immediately
    try:
        print("Server attempting to close...")
        await websocket.close(1000, "Completed")
        print("Server close success")
    except Exception as e:
        print("Server close threw:", type(e))

