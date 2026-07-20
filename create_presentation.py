"""
Generate the ARIA (Analytical Risk Intelligence Agent) demo presentation.
Run: python create_presentation.py
Output: ARIA_Platform_Demo.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Brand colors ─────────────────────────────────────────────────────
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT_BLUE = RGBColor(0x00, 0x96, 0xD6)
ACCENT_GREEN = RGBColor(0x00, 0xC9, 0x7B)
ACCENT_ORANGE = RGBColor(0xFF, 0x8C, 0x42)
ACCENT_RED = RGBColor(0xE8, 0x4D, 0x4D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MEDIUM_GRAY = RGBColor(0x66, 0x66, 0x66)
SUBTLE_BG = RGBColor(0xF5, 0xF7, 0xFA)
CARD_BG = RGBColor(0xE8, 0xEE, 0xF4)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height


def add_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color=None, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color or CARD_BG
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=DARK_GRAY, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_slide_content(slide, left, top, width, height, items, font_size=16,
                              color=DARK_GRAY, spacing=Pt(8)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0
    return txBox


def add_flow_box(slide, left, top, width, height, text, fill_color=ACCENT_BLUE, font_size=11):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.font.name = "Calibri"
    shape.text_frame.margin_left = Pt(4)
    shape.text_frame.margin_right = Pt(4)
    shape.text_frame.margin_top = Pt(4)
    shape.text_frame.margin_bottom = Pt(4)
    return shape


def add_arrow(slide, left, top, width=Inches(0.4), height=Inches(0.01)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = MEDIUM_GRAY
    shape.line.fill.background()
    return shape


def add_down_arrow(slide, left, top, width=Inches(0.01), height=Inches(0.3)):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = MEDIUM_GRAY
    shape.line.fill.background()
    return shape


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, DARK_BG)

add_text_box(slide, Inches(1.5), Inches(1.5), Inches(10), Inches(1.2),
             "ARIA — Analytical Risk Intelligence Agent", font_size=44, color=WHITE, bold=True)
add_text_box(slide, Inches(1.5), Inches(2.7), Inches(10), Inches(0.8),
             "Federated RAG Platform for Enterprise Compliance & Risk Analytics",
             font_size=22, color=ACCENT_BLUE)

items = [
    "Deterministic metric computation  |  Multi-strategy query routing",
    "LangGraph orchestration  |  Production observability  |  Regulator-grade audit trails",
]
add_bullet_slide_content(slide, Inches(1.5), Inches(4.0), Inches(10), Inches(1.5),
                         items, font_size=16, color=LIGHT_GRAY)

add_text_box(slide, Inches(1.5), Inches(6.0), Inches(10), Inches(0.5),
             "Pranab Akhoury  |  June 2026", font_size=14, color=MEDIUM_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — The Problem
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "The Problem: Why Enterprise RAG Fails", font_size=32, color=DARK_GRAY, bold=True)

# Left column — typical RAG
add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.2), fill_color=RGBColor(0xFF, 0xF0, 0xF0))
add_text_box(slide, Inches(1.0), Inches(1.6), Inches(5), Inches(0.5),
             "Typical RAG System", font_size=20, color=ACCENT_RED, bold=True)
items = [
    "User -> LLM -> Vector Search -> Answer",
    "",
    "Single path for all query types",
    "LLM generates SQL freely (hallucination risk)",
    "No validation — 'generate and pray'",
    "No audit trail for regulators",
    "No metric governance or versioning",
    "Breaks at 10+ tables / 100+ documents",
]
add_bullet_slide_content(slide, Inches(1.0), Inches(2.2), Inches(5), Inches(4),
                         items, font_size=15, color=DARK_GRAY)

# Right column — this platform
add_shape(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(5.2), fill_color=RGBColor(0xF0, 0xFF, 0xF0))
add_text_box(slide, Inches(7.2), Inches(1.6), Inches(5), Inches(0.5),
             "This Platform", font_size=20, color=ACCENT_GREEN, bold=True)
items = [
    "Router -> Strategy -> SQL + Docs -> Review -> Answer",
    "",
    "5 routing strategies based on query type",
    "Deterministic SQL for known metrics (zero LLM)",
    "5 validators + LLM reviewer + reflection loop",
    "PostgreSQL audit trail (mandatory in prod)",
    "Versioned metrics with golden test suite",
    "Tested at 50 tables / 100s of documents",
]
add_bullet_slide_content(slide, Inches(7.2), Inches(2.2), Inches(5), Inches(4),
                         items, font_size=15, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — Architecture Overview
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.3), Inches(11), Inches(0.7),
             "Architecture: LangGraph Workflow", font_size=32, color=DARK_GRAY, bold=True)

# Flow diagram
y = Inches(1.3)
bw, bh = Inches(1.6), Inches(0.55)
gap = Inches(0.15)

# Row 1: Cache -> Context Resolver -> Router -> Clarify
x = Inches(0.5)
add_flow_box(slide, x, y, bw, bh, "Cache Check", ACCENT_BLUE)
x += bw + gap
add_arrow(slide, x, y + Inches(0.22))
x += Inches(0.5)
add_flow_box(slide, x, y, bw, bh, "Context\nResolver", ACCENT_BLUE)
x += bw + gap
add_arrow(slide, x, y + Inches(0.22))
x += Inches(0.5)
add_flow_box(slide, x, y, bw, bh, "Router", ACCENT_ORANGE)
x += bw + gap
add_arrow(slide, x, y + Inches(0.22))
x += Inches(0.5)
add_flow_box(slide, x, y, bw, bh, "Clarify", ACCENT_BLUE)
x += bw + gap
add_arrow(slide, x, y + Inches(0.22))
x += Inches(0.5)
add_flow_box(slide, x, y, bw, bh, "Metric\nResolver", ACCENT_GREEN)

# Row 2: Branching paths
y2 = Inches(2.5)
add_down_arrow(slide, Inches(9.5), y + bh, height=Inches(0.4))

# SQL Path
add_flow_box(slide, Inches(0.5), y2, Inches(2.2), bh, "Compiled Metric\n(Deterministic SQL)", ACCENT_GREEN)
add_text_box(slide, Inches(0.5), y2 + bh + Inches(0.05), Inches(2.2), Inches(0.3),
             "Zero LLM calls", font_size=10, color=ACCENT_GREEN, alignment=PP_ALIGN.CENTER)

add_flow_box(slide, Inches(3.0), y2, Inches(2.2), bh, "LLM SQL Path\n(Ad-hoc Queries)", ACCENT_ORANGE)
add_text_box(slide, Inches(3.0), y2 + bh + Inches(0.05), Inches(2.2), Inches(0.3),
             "Retry + Validate", font_size=10, color=ACCENT_ORANGE, alignment=PP_ALIGN.CENTER)

add_flow_box(slide, Inches(5.5), y2, Inches(2.2), bh, "Vector Retrieval\n(Doc Search)", ACCENT_BLUE)
add_text_box(slide, Inches(5.5), y2 + bh + Inches(0.05), Inches(2.2), Inches(0.3),
             "Metadata Filtered", font_size=10, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)

add_flow_box(slide, Inches(8.0), y2, Inches(2.0), bh, "Extract Node\n(Chain Step)", ACCENT_BLUE)

# Row 3: Answer + Review
y3 = Inches(4.0)
add_flow_box(slide, Inches(3.0), y3, Inches(2.5), bh, "Answer Generator", ACCENT_BLUE)
add_arrow(slide, Inches(5.6), y3 + Inches(0.22))
add_flow_box(slide, Inches(6.2), y3, Inches(2.5), bh, "Reviewer\n(LLM + 5 Validators)", ACCENT_ORANGE)
add_arrow(slide, Inches(8.8), y3 + Inches(0.22))
add_flow_box(slide, Inches(9.4), y3, Inches(2.0), bh, "Cache Write", ACCENT_GREEN)

# Reflection arrow
add_text_box(slide, Inches(5.8), y3 + bh + Inches(0.05), Inches(3), Inches(0.3),
             "Score < 7.0 -> Reflection Loop (max 2 attempts)", font_size=10, color=ACCENT_RED)

# Routing strategies box
y4 = Inches(5.2)
add_shape(slide, Inches(0.5), y4, Inches(12), Inches(1.8), fill_color=SUBTLE_BG)
add_text_box(slide, Inches(0.7), y4 + Inches(0.1), Inches(11), Inches(0.4),
             "5 Routing Strategies", font_size=18, color=DARK_GRAY, bold=True)

strategies = [
    ("SQL_ONLY", "Pure database queries — counts, rankings, aggregations", ACCENT_BLUE),
    ("DOCS_ONLY", "Policy lookups, definitions, explanations", ACCENT_BLUE),
    ("DOCS_THEN_SQL", "Get formula from docs, then query data", ACCENT_ORANGE),
    ("SQL_THEN_DOCS", "Get data first, then find related policy", ACCENT_ORANGE),
    ("PARALLEL", "Independent data + docs fetch", ACCENT_GREEN),
]
for i, (name, desc, color) in enumerate(strategies):
    col = i % 3
    row = i // 3
    x = Inches(0.7) + col * Inches(4.0)
    yy = y4 + Inches(0.55) + row * Inches(0.55)
    add_text_box(slide, x, yy, Inches(1.5), Inches(0.4), name, font_size=12, color=color, bold=True)
    add_text_box(slide, x + Inches(1.6), yy, Inches(2.3), Inches(0.4), desc, font_size=11, color=MEDIUM_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — Deterministic Metric Compiler
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Deterministic Metric Compiler: Zero LLM Risk", font_size=32, color=DARK_GRAY, bold=True)

add_text_box(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(0.5),
             "Registered metrics bypass the LLM entirely. Same input always produces same SQL.",
             font_size=16, color=MEDIUM_GRAY)

# Before/After comparison
add_shape(slide, Inches(0.5), Inches(1.8), Inches(5.8), Inches(2.5), fill_color=RGBColor(0xFF, 0xF0, 0xF0))
add_text_box(slide, Inches(0.7), Inches(1.9), Inches(5.4), Inches(0.4),
             "BEFORE: LLM Interprets Metric", font_size=16, color=ACCENT_RED, bold=True)
before_items = [
    "Metric Definition -> Prompt -> LLM -> SQL",
    "LLM can misread formula, omit filters, use wrong JOINs",
    "Same question can produce different SQL each time",
    "No guarantee formula is followed correctly",
]
add_bullet_slide_content(slide, Inches(0.7), Inches(2.4), Inches(5.4), Inches(1.8),
                         before_items, font_size=13, color=DARK_GRAY, spacing=Pt(4))

add_shape(slide, Inches(7.0), Inches(1.8), Inches(5.8), Inches(2.5), fill_color=RGBColor(0xF0, 0xFF, 0xF0))
add_text_box(slide, Inches(7.2), Inches(1.9), Inches(5.4), Inches(0.4),
             "AFTER: Deterministic Compiler", font_size=16, color=ACCENT_GREEN, bold=True)
after_items = [
    "Metric Definition -> Compiler -> Verified SQL",
    "SQL assembled from structured clauses (SELECT, WHERE, GROUP BY)",
    "Same input = same SQL, every time",
    "Parameters sanitized, row limits enforced automatically",
]
add_bullet_slide_content(slide, Inches(7.2), Inches(2.4), Inches(5.4), Inches(1.8),
                         after_items, font_size=13, color=DARK_GRAY, spacing=Pt(4))

# Metrics table
add_text_box(slide, Inches(0.8), Inches(4.6), Inches(11), Inches(0.5),
             "7 Registered Metrics (all compiled deterministically)", font_size=18, color=DARK_GRAY, bold=True)

metrics = [
    ("Compliance Effectiveness Score", "%", "clause"),
    ("Violation Severity Distribution", "count", "clause"),
    ("Audit Finding Resolution Rate", "%", "clause"),
    ("Regulatory Exposure Index", "USD", "clause"),
    ("Department Risk Score", "score", "template"),
    ("Control Coverage Ratio", "%", "clause"),
    ("Mean Time to Resolution", "days", "clause"),
]
for i, (name, unit, mode) in enumerate(metrics):
    col = i % 2
    row = i // 2
    x = Inches(0.8) + col * Inches(6.2)
    yy = Inches(5.2) + row * Inches(0.45)
    add_text_box(slide, x, yy, Inches(3.5), Inches(0.4), name, font_size=13, color=DARK_GRAY, bold=True)
    add_text_box(slide, x + Inches(3.5), yy, Inches(0.8), Inches(0.4), unit, font_size=12, color=MEDIUM_GRAY)
    add_text_box(slide, x + Inches(4.4), yy, Inches(1.0), Inches(0.4), mode, font_size=12, color=ACCENT_GREEN)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — Correctness & Validation
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Correctness: 5 Deterministic Validators + LLM Reviewer", font_size=32, color=DARK_GRAY, bold=True)

validators = [
    ("SQL Safety", "Blocks DROP, DELETE, injection, unbalanced parens", "cap=0.0"),
    ("Result Sanity", "Flags empty results, suspiciously large numbers", "cap=5.0"),
    ("Answer Grounding", "Ensures answer has supporting data for its claims", "cap=3.0"),
    ("Number Grounding", "Checks numeric claims appear in SQL result", "cap=5.0"),
    ("Metric Citation", "Compiled metrics must be referenced in answer", "cap=6.0"),
]

for i, (name, desc, cap) in enumerate(validators):
    yy = Inches(1.5) + i * Inches(0.9)
    add_shape(slide, Inches(0.8), yy, Inches(7.5), Inches(0.75), fill_color=SUBTLE_BG)
    add_text_box(slide, Inches(1.0), yy + Inches(0.05), Inches(2.5), Inches(0.35),
                 name, font_size=16, color=ACCENT_BLUE, bold=True)
    add_text_box(slide, Inches(1.0), yy + Inches(0.38), Inches(6), Inches(0.3),
                 desc, font_size=12, color=MEDIUM_GRAY)
    add_text_box(slide, Inches(6.8), yy + Inches(0.1), Inches(1.3), Inches(0.3),
                 cap, font_size=11, color=ACCENT_RED, bold=True)

# Confidence scoring
add_shape(slide, Inches(9.0), Inches(1.5), Inches(3.8), Inches(4.5), fill_color=CARD_BG)
add_text_box(slide, Inches(9.2), Inches(1.6), Inches(3.4), Inches(0.5),
             "Confidence Scoring", font_size=18, color=DARK_GRAY, bold=True)

conf_items = [
    "HIGH: Compiled metric +",
    "  score >= 8.0 + no failures",
    "",
    "MEDIUM: LLM SQL path +",
    "  good score + supporting data",
    "",
    "LOW: Retries needed or",
    "  validator failures detected",
    "",
    "Exposed in every API response",
    "for enterprise audit visibility",
]
add_bullet_slide_content(slide, Inches(9.2), Inches(2.2), Inches(3.4), Inches(3.5),
                         conf_items, font_size=12, color=DARK_GRAY, spacing=Pt(2))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — Scalability
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Scalability: 50 Tables, 100s of Documents", font_size=32, color=DARK_GRAY, bold=True)

scale_features = [
    ("Per-Table Catalog", "One JSON per table + lightweight index. On-demand loading. Add a table = 1 file.", ACCENT_BLUE),
    ("Slim Table Selection", "1 line per table in LLM prompt (~2K tokens at 50 tables vs ~15K before).", ACCENT_BLUE),
    ("No Metric Dump", "Resolved metrics use compiler. Ad-hoc queries get clean schema only.", ACCENT_GREEN),
    ("Metadata-Filtered Docs", "PGVector JSONB filtering by doc_type/category. k=6 with route-aware filters.", ACCENT_GREEN),
    ("Synonym Fallback", "12 compliance synonyms (exposure->RISK_EVENTS, penalty->VIOLATIONS, etc.).", ACCENT_ORANGE),
    ("Model Fallback Chain", "Primary LLM + fallback model. Circuit breaker prevents cascading failures.", ACCENT_ORANGE),
]

for i, (title, desc, color) in enumerate(scale_features):
    col = i % 2
    row = i // 2
    x = Inches(0.5) + col * Inches(6.3)
    yy = Inches(1.5) + row * Inches(1.6)
    add_shape(slide, x, yy, Inches(5.8), Inches(1.35), fill_color=SUBTLE_BG)
    add_text_box(slide, x + Inches(0.2), yy + Inches(0.1), Inches(5.4), Inches(0.4),
                 title, font_size=16, color=color, bold=True)
    add_text_box(slide, x + Inches(0.2), yy + Inches(0.55), Inches(5.4), Inches(0.7),
                 desc, font_size=13, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — Enterprise Security & Observability
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Enterprise: Security, Observability, Audit", font_size=32, color=DARK_GRAY, bold=True)

# Security column
add_shape(slide, Inches(0.5), Inches(1.3), Inches(3.8), Inches(5.5), fill_color=SUBTLE_BG)
add_text_box(slide, Inches(0.7), Inches(1.4), Inches(3.4), Inches(0.5),
             "Security", font_size=20, color=ACCENT_RED, bold=True)
sec_items = [
    "API key authentication",
    "Rate limiting (per-key + per-IP)",
    "Prompt injection detection",
    "SQL allowlist (SELECT only)",
    "3-layer SQL validation",
    "PII redaction (email, SSN, phone)",
    "DLP scan on every answer",
    "Oracle package blocklist",
]
add_bullet_slide_content(slide, Inches(0.7), Inches(2.0), Inches(3.4), Inches(4.5),
                         sec_items, font_size=13, color=DARK_GRAY, spacing=Pt(6))

# Observability column
add_shape(slide, Inches(4.7), Inches(1.3), Inches(3.8), Inches(5.5), fill_color=SUBTLE_BG)
add_text_box(slide, Inches(4.9), Inches(1.4), Inches(3.4), Inches(0.5),
             "Observability", font_size=20, color=ACCENT_BLUE, bold=True)
obs_items = [
    "OpenTelemetry distributed tracing",
    "Prometheus metrics (20+ counters)",
    "Grafana dashboards",
    "Per-node latency histograms",
    "LLM token + cost tracking",
    "Circuit breaker state gauge",
    "Review score distribution",
    "Structured JSON logging",
]
add_bullet_slide_content(slide, Inches(4.9), Inches(2.0), Inches(3.4), Inches(4.5),
                         obs_items, font_size=13, color=DARK_GRAY, spacing=Pt(6))

# Audit column
add_shape(slide, Inches(8.9), Inches(1.3), Inches(3.8), Inches(5.5), fill_color=SUBTLE_BG)
add_text_box(slide, Inches(9.1), Inches(1.4), Inches(3.4), Inches(0.5),
             "Audit & Governance", font_size=20, color=ACCENT_GREEN, bold=True)
audit_items = [
    "PostgreSQL audit trail (mandatory)",
    "Full state history per session",
    "/audit/{session_id} endpoint",
    "Metric versioning (id@version)",
    "Execution metadata in response",
    "Confidence scoring (H/M/L)",
    "MemorySaver blocked in prod",
    "Deterministic SQL = reproducible",
]
add_bullet_slide_content(slide, Inches(9.1), Inches(2.0), Inches(3.4), Inches(4.5),
                         audit_items, font_size=13, color=DARK_GRAY, spacing=Pt(6))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — Test Coverage & Evaluation
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Test Coverage: 379 Tests, 6 Test Suites", font_size=32, color=DARK_GRAY, bold=True)

suites = [
    ("test_nodes.py", "~60", "Every workflow node in isolation"),
    ("test_metric_compiler.py", "39", "Resolver -> Compiler -> SQL for all 7 metrics"),
    ("test_validators.py", "39", "All 5 deterministic validators"),
    ("test_eval.py", "56", "21 golden cases x routing + quality + confidence"),
    ("test_workflow.py", "10", "Full graph integration, all 5 routes"),
    ("test_metrics.py", "~140", "Registry, loader, resolver, LLM resolver, compiler"),
    ("test_security.py", "~35", "Auth, rate limits, injection, PII, DLP"),
]

add_shape(slide, Inches(0.5), Inches(1.3), Inches(12), Inches(0.55), fill_color=ACCENT_BLUE)
add_text_box(slide, Inches(0.7), Inches(1.35), Inches(3), Inches(0.4),
             "Suite", font_size=14, color=WHITE, bold=True)
add_text_box(slide, Inches(3.8), Inches(1.35), Inches(1), Inches(0.4),
             "Tests", font_size=14, color=WHITE, bold=True)
add_text_box(slide, Inches(5.0), Inches(1.35), Inches(7), Inches(0.4),
             "Coverage", font_size=14, color=WHITE, bold=True)

for i, (suite, count, coverage) in enumerate(suites):
    yy = Inches(1.9) + i * Inches(0.55)
    bg = SUBTLE_BG if i % 2 == 0 else WHITE
    add_shape(slide, Inches(0.5), yy, Inches(12), Inches(0.5), fill_color=bg)
    add_text_box(slide, Inches(0.7), yy + Inches(0.05), Inches(3), Inches(0.4),
                 suite, font_size=13, color=DARK_GRAY, bold=True)
    add_text_box(slide, Inches(3.8), yy + Inches(0.05), Inches(1), Inches(0.4),
                 count, font_size=13, color=ACCENT_BLUE, bold=True)
    add_text_box(slide, Inches(5.0), yy + Inches(0.05), Inches(7), Inches(0.4),
                 coverage, font_size=13, color=MEDIUM_GRAY)

# CI badge
add_shape(slide, Inches(0.5), Inches(5.9), Inches(12), Inches(1.0), fill_color=CARD_BG)
add_text_box(slide, Inches(0.7), Inches(6.0), Inches(11), Inches(0.8),
             "CI Pipeline: Lint (ruff) -> All Tests (pytest) -> Docker Build -> Verify\n"
             "Runs on every push to main and develop. Coverage threshold: 60%.",
             font_size=14, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — API Response
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "API Response: Full Transparency", font_size=32, color=WHITE, bold=True)

json_text = """{
    "answer": "The compliance effectiveness score is 85%,
               rated as Good (80-95% threshold).",
    "cache_hit": false,
    "session_id": "abc-123",
    "trace_id": "tr-9f8e7d",
    "metadata": {
        "route": "sql_only",
        "resolved_metric": "compliance_effectiveness_score",
        "metric_version": "1.0",
        "compiled": true,
        "confidence": "high",
        "review_score": 9.0
    }
}"""

add_shape(slide, Inches(1.0), Inches(1.3), Inches(7.5), Inches(5.5), fill_color=RGBColor(0x22, 0x22, 0x3E))
add_text_box(slide, Inches(1.3), Inches(1.5), Inches(7), Inches(5),
             json_text, font_size=15, color=ACCENT_GREEN, font_name="Consolas")

# Annotations
annotations = [
    (Inches(1.7), "Natural language answer grounded in data"),
    (Inches(3.6), "Which routing strategy was used"),
    (Inches(4.1), "Which registered metric matched"),
    (Inches(4.6), "Deterministic compiler was used (no LLM for SQL)"),
    (Inches(5.1), "How much to trust this answer"),
    (Inches(5.6), "LLM + validator review score"),
]

for yy, text in annotations:
    add_text_box(slide, Inches(9.0), yy, Inches(3.8), Inches(0.4),
                 text, font_size=12, color=LIGHT_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 10 — Tech Stack
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Technology Stack", font_size=32, color=DARK_GRAY, bold=True)

stack = [
    ("Orchestration", "LangGraph (deterministic workflow, not agent loops)", ACCENT_BLUE),
    ("LLM", "Groq (llama-3.3-70b) + fallback (llama-3.1-8b)", ACCENT_BLUE),
    ("Database", "Oracle (structured data) + PostgreSQL (vectors + audit)", ACCENT_GREEN),
    ("Vector Store", "PGVector with JSONB metadata filtering", ACCENT_GREEN),
    ("Caching", "Redis (answers + conversation history)", ACCENT_ORANGE),
    ("API", "FastAPI with OpenAPI docs, rate limiting, API key auth", ACCENT_ORANGE),
    ("Observability", "OpenTelemetry + Prometheus + Grafana", ACCENT_BLUE),
    ("CI/CD", "GitHub Actions: lint -> test -> Docker build", ACCENT_BLUE),
    ("Container", "Multi-stage Docker, non-root, health checks, resource limits", ACCENT_GREEN),
    ("Embeddings", "HuggingFace all-MiniLM-L6-v2", ACCENT_GREEN),
]

for i, (category, detail, color) in enumerate(stack):
    col = i % 2
    row = i // 2
    x = Inches(0.5) + col * Inches(6.3)
    yy = Inches(1.3) + row * Inches(1.1)
    add_shape(slide, x, yy, Inches(5.8), Inches(0.9), fill_color=SUBTLE_BG)
    add_text_box(slide, x + Inches(0.2), yy + Inches(0.05), Inches(2.2), Inches(0.4),
                 category, font_size=14, color=color, bold=True)
    add_text_box(slide, x + Inches(0.2), yy + Inches(0.45), Inches(5.4), Inches(0.4),
                 detail, font_size=12, color=DARK_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 11 — Scores
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BG)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Platform Scores", font_size=32, color=WHITE, bold=True)

scores = [
    ("Architecture", "9.5", "Dual-path SQL, LangGraph workflow, per-table catalog, model fallback"),
    ("Correctness", "9.0", "Deterministic compiler, 5 validators, confidence scoring, golden tests"),
    ("Production Readiness", "9.0", "Enforced audit trail, enriched cache, fallback chain, CI/CD"),
    ("Scalability", "9.0", "Slim prompts, on-demand catalog, metadata-filtered docs, synonyms"),
    ("Test Coverage", "9.5", "379 tests, 6 suites, eval harness with 21 golden cases"),
]

for i, (category, score, detail) in enumerate(scores):
    yy = Inches(1.5) + i * Inches(1.1)
    add_shape(slide, Inches(0.8), yy, Inches(11.5), Inches(0.9), fill_color=RGBColor(0x22, 0x22, 0x3E))

    score_color = ACCENT_GREEN if float(score) >= 9.0 else ACCENT_ORANGE
    add_text_box(slide, Inches(1.0), yy + Inches(0.05), Inches(1.2), Inches(0.4),
                 score, font_size=28, color=score_color, bold=True)
    add_text_box(slide, Inches(2.3), yy + Inches(0.05), Inches(3), Inches(0.4),
                 category, font_size=18, color=WHITE, bold=True)
    add_text_box(slide, Inches(2.3), yy + Inches(0.48), Inches(9.5), Inches(0.4),
                 detail, font_size=12, color=LIGHT_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 12 — Thank You / Contact
# ═══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BG)

add_text_box(slide, Inches(1.5), Inches(2.0), Inches(10), Inches(1.2),
             "ARIA — Analytical Risk Intelligence Agent", font_size=40, color=WHITE, bold=True,
             alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1.5), Inches(3.2), Inches(10), Inches(0.8),
             "Enterprise-Grade Federated RAG for Compliance & Risk",
             font_size=20, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1.5), Inches(4.5), Inches(10), Inches(1.5),
             "Pranab Akhoury\npranab.akhoury@gmail.com\nGitHub: intelligence-ai-engine",
             font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────
output_path = "ARIA_Platform_Demo.pptx"
prs.save(output_path)
print(f"Presentation saved: {output_path}")
print(f"  Slides: {len(prs.slides)}")
