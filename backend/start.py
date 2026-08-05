import os
import uvicorn

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )