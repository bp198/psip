"""
PSIP API — entry point.

Run locally:
    python main.py

Or with uvicorn directly:
    uvicorn psip.api:app --host 0.0.0.0 --port 8000 --reload

Interactive docs once running:
    http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "psip.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["psip"],
    )
