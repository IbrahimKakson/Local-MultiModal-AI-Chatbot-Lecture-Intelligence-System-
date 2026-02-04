from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="Lecture Intelligence System")

app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
