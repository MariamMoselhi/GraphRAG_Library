# HoloLearn-RAG/run_server.py

import sys
import asyncio
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # always loads D:\HoloLearn-RAG\.env

async def main():
    config = uvicorn.Config(
        "api_server:app",
        host  = "127.0.0.1",
        port  = 8002,
        loop  = "asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())