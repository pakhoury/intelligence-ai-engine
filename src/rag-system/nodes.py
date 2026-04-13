import os
from config import llm, db_connector, redis_client, load_prompt
from datetime import timedelta
from typing import Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_vector_store():
    """Lazy-load the PGVector store for retrieval."""
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_db = os.getenv("POSTGRES_DB", "rag_db")
    connection = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}"

    return PGVector(
        collection_name="compliance_docs",
        connection_string=connection,
        embedding_function=_get_embeddings(),
        use_jsonb=True,
    )


def cache_check(state: Dict):
    question = state["question"]
    try:
        cached = redis_client.get(question)
        if cached:
            print(f"[cache_check] Cache HIT")
            return {"cache_hit": True, "final_answer": cached, "review_score": 10.0}
    except Exception:
        pass
    print(f"[cache_check] Cache MISS")
    return {"cache_hit": False}


def router_node(state: Dict):
    prompt = load_prompt("router.txt").format(question=state["question"])
    response = llm.invoke(prompt)
    content = response.content.strip().upper()

    if "BOTH" in content:
        route = "both"
    elif "SQL" in content:
        route = "sql"
    else:
        route = "documents"

    print(f"[router] Route decided: {route}")
    return {"route": route}


def clarification_node(state: Dict):
    prompt = load_prompt("clarification.txt").format(question=state["question"])
    response = llm.invoke(prompt)
    if "NO_CLARIFICATION_NEEDED" in response.content.upper():
        print(f"[clarify] No clarification needed")
        return {"needs_clarification": False}
    print(f"[clarify] Clarification needed")
    return {
        "needs_clarification": True,
        "final_answer": response.content.strip(),
        "review_score": 10.0
    }


def sql_path(state: Dict):
    question = state["question"]
    print(f"[sql_path] Generating SQL for: {question}")

    tables = db_connector.get_relevant_tables(question)
    schema = db_connector.get_table_schema(tables)

    prompt = load_prompt("sql_agent.txt").format(
        question=question,
        schema_description=schema
    )
    sql = llm.invoke(prompt).content.strip()
    # Strip markdown code fences if present
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    print(f"[sql_path] Generated SQL: {sql}")

    sql_result = db_connector.execute_query(sql)
    print(f"[sql_path] Result: {sql_result[:200]}")

    return {"sql_result": sql_result}


def vector_retrieval(state: Dict):
    question = state["question"]
    print(f"[vector_retrieval] Searching documents for: {question}")

    try:
        store = _get_vector_store()
        results = store.similarity_search(question, k=4)
        docs = []
        for doc in results:
            source = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
            docs.append(f"[{source}]: {doc.page_content}")
        print(f"[vector_retrieval] Found {len(docs)} relevant chunks")
        return {"retrieved_docs": docs}
    except Exception as e:
        print(f"[vector_retrieval] Error: {e}")
        return {"retrieved_docs": []}


def answer_generator(state: Dict):
    route = state.get("route", "documents")
    print(f"[answer_generator] Generating answer (route={route})")

    prompt = load_prompt("final_answer.txt").format(
        question=state["question"],
        sql_result=state.get("sql_result", "N/A - no SQL data retrieved"),
        context="\n".join(state.get("retrieved_docs", [])) or "N/A - no documents retrieved"
    )
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


def reviewer_node(state: Dict):
    prompt = load_prompt("reviewer.txt").format(
        question=state["question"],
        sql_result=state.get("sql_result", ""),
        docs=state.get("retrieved_docs", ""),
        answer=state.get("final_answer", "")
    )
    response = llm.invoke(prompt)
    score = 7.0
    if "Score:" in response.content:
        try:
            score = float(response.content.split("Score:")[1].strip().split()[0])
        except Exception:
            pass
    print(f"[reviewer] Score: {score}")
    return {"review_score": score}


def cache_write(state: Dict):
    if state.get("review_score", 0) >= 7.0 and state.get("final_answer"):
        try:
            redis_client.set(state["question"], state["final_answer"], ex=timedelta(hours=24))
            print(f"[cache_write] Cached answer")
        except Exception:
            pass
    return state
