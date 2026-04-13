from langgraph.graph import StateGraph, END
from typing import TypedDict
from nodes import (
    cache_check, router_node, clarification_node, sql_path,
    vector_retrieval, answer_generator, reviewer_node, cache_write
)

class AgentState(TypedDict, total=False):
    question: str
    cache_hit: bool
    final_answer: str
    review_score: float
    route: str  # "sql", "documents", or "both"
    needs_clarification: bool
    sql_result: str
    retrieved_docs: list

workflow = StateGraph(AgentState)

workflow.add_node("cache_check", cache_check)
workflow.add_node("router", router_node)
workflow.add_node("clarify", clarification_node)
workflow.add_node("sql_path", sql_path)
workflow.add_node("vector_retrieval", vector_retrieval)
workflow.add_node("answer_generator", answer_generator)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("cache_write", cache_write)

workflow.set_entry_point("cache_check")

# Cache hit → END, miss → router
workflow.add_conditional_edges(
    "cache_check",
    lambda s: END if s.get("cache_hit") else "router",
    {END: END, "router": "router"}
)

workflow.add_edge("router", "clarify")

# After clarification: route based on question type
def after_clarify(s):
    if s.get("needs_clarification", False):
        return END
    route = s.get("route", "documents")
    if route == "sql":
        return "sql_path"
    elif route == "both":
        return "sql_path"  # sql first, then vector_retrieval
    return "vector_retrieval"

workflow.add_conditional_edges(
    "clarify",
    after_clarify,
    {"sql_path": "sql_path", "vector_retrieval": "vector_retrieval", END: END}
)

# After sql_path: if BOTH, also do vector retrieval; otherwise go to answer
def after_sql(s):
    if s.get("route") == "both":
        return "vector_retrieval"
    return "answer_generator"

workflow.add_conditional_edges(
    "sql_path",
    after_sql,
    {"vector_retrieval": "vector_retrieval", "answer_generator": "answer_generator"}
)

workflow.add_edge("vector_retrieval", "answer_generator")
workflow.add_edge("answer_generator", "reviewer")
workflow.add_edge("reviewer", "cache_write")
workflow.add_edge("cache_write", END)

app = workflow.compile()
