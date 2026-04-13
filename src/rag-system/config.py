from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_groq import ChatGroq
from database.oracle import OracleConnector
import redis

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1200
)

db_connector = OracleConnector(llm=llm)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
    socket_timeout=5
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(filename: str) -> str:
    try:
        with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: prompts/{filename} not found")
        return ""