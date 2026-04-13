from fastapi import FastAPI
from pydantic import BaseModel
from workflow import app as workflow_app

app = FastAPI(title="Compliance RAG System")

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(request: Query):
    result = await workflow_app.ainvoke({"question": request.question})
    return {
        "answer": result.get("final_answer", "Sorry, I could not generate an answer."),
        "cache_hit": result.get("cache_hit", False)
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}