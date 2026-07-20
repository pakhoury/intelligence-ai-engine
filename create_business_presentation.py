"""
Generate a business/leadership-focused demo presentation.
Run: python create_business_presentation.py
Output: Intelligence_AI_Engine_Business_Demo.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Brand colors ─────────────────────────────────────────────────────
NAVY = RGBColor(0x0A, 0x1A, 0x3A)
DEEP_BLUE = RGBColor(0x12, 0x2B, 0x5C)
ACCENT = RGBColor(0x00, 0x8C, 0xD2)
GREEN = RGBColor(0x00, 0xB3, 0x6B)
AMBER = RGBColor(0xFF, 0xA0, 0x26)
RED = RGBColor(0xE0, 0x3E, 0x3E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE = RGBColor(0xF8, 0xF9, 0xFA)
LIGHT_GRAY = RGBColor(0xBB, 0xBB, 0xBB)
MID_GRAY = RGBColor(0x88, 0x88, 0x88)
DARK_TEXT = RGBColor(0x2D, 0x2D, 0x2D)
SOFT_BG = RGBColor(0xEE, 0xF2, 0xF7)
CARD_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN_BG = RGBColor(0xE8, 0xF8, 0xF0)
RED_BG = RGBColor(0xFD, 0xF0, 0xF0)
BLUE_BG = RGBColor(0xE8, 0xF0, 0xFE)
AMBER_BG = RGBColor(0xFE, 0xF7, 0xE8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def shape(slide, left, top, w, h, fill=CARD_WHITE, radius=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    return s


def rect(slide, left, top, w, h, fill=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    return s


def txt(slide, left, top, w, h, text, size=18, color=DARK_TEXT,
        bold=False, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(left, top, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font
    p.alignment = align
    return tb


def bullets(slide, left, top, w, h, items, size=16, color=DARK_TEXT,
            spacing=Pt(10), bold_first=False):
    tb = slide.shapes.add_textbox(left, top, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        if bold_first and ":" in item:
            parts = item.split(":", 1)
            run = p.add_run()
            run.text = parts[0] + ":"
            run.font.bold = True
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.name = "Calibri"
            run2 = p.add_run()
            run2.text = parts[1]
            run2.font.size = Pt(size)
            run2.font.color.rgb = color
            run2.font.name = "Calibri"
        else:
            p.text = item
    return tb


def stat_card(slide, left, top, w, h, number, label, accent=ACCENT, bg=CARD_WHITE):
    shape(slide, left, top, w, h, fill=bg)
    rect(slide, left, top, w, Pt(4), fill=accent)
    txt(slide, left + Inches(0.2), top + Inches(0.2), w - Inches(0.4), Inches(0.7),
        number, size=36, color=accent, bold=True, align=PP_ALIGN.CENTER)
    txt(slide, left + Inches(0.2), top + Inches(0.85), w - Inches(0.4), Inches(0.5),
        label, size=13, color=MID_GRAY, align=PP_ALIGN.CENTER)


def icon_card(slide, left, top, w, h, title, desc, accent=ACCENT, bg=CARD_WHITE):
    shape(slide, left, top, w, h, fill=bg)
    rect(slide, left, top, w, Pt(4), fill=accent)
    txt(slide, left + Inches(0.25), top + Inches(0.2), w - Inches(0.5), Inches(0.4),
        title, size=16, color=accent, bold=True)
    txt(slide, left + Inches(0.25), top + Inches(0.65), w - Inches(0.5), h - Inches(0.8),
        desc, size=12, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, NAVY)

rect(sl, Inches(0), Inches(0), Inches(0.15), Inches(7.5), fill=ACCENT)

txt(sl, Inches(1.2), Inches(1.8), Inches(10), Inches(1.0),
    "ARIA — Analytical Risk Intelligence Agent", size=48, color=WHITE, bold=True)
txt(sl, Inches(1.2), Inches(3.0), Inches(10), Inches(0.6),
    "AI-Powered Compliance & Risk Analytics Platform", size=24, color=ACCENT)

txt(sl, Inches(1.2), Inches(4.2), Inches(9), Inches(1.2),
    "Ask questions in plain English. Get answers grounded in your data and policies.\n"
    "Every answer is validated, auditable, and regulator-ready.",
    size=16, color=LIGHT_GRAY)

txt(sl, Inches(1.2), Inches(6.2), Inches(10), Inches(0.4),
    "Pranab Akhoury  |  June 2026", size=14, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — The Challenge
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "The Challenge", size=34, color=NAVY, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.5),
    "Compliance teams spend hours getting answers that should take seconds",
    size=17, color=MID_GRAY)

challenges = [
    ("Manual Report Building",
     "Analysts write SQL queries by hand, cross-reference policy documents, "
     "and compile answers into spreadsheets. A single question can take 30-60 minutes."),
    ("Inconsistent Answers",
     "Two analysts answering the same question may use different formulas, "
     "different date ranges, or different filters — producing different numbers."),
    ("Audit Trail Gaps",
     "When a regulator asks \"how did you calculate this number?\", the answer is often "
     "buried in email threads and local spreadsheets. Reproducing it takes days."),
    ("Scaling Pain",
     "As tables, documents, and regulatory requirements grow, manual processes break. "
     "Adding a new metric means updating multiple reports and training analysts."),
]

for i, (title, desc) in enumerate(challenges):
    col = i % 2
    row = i // 2
    x = Inches(0.5) + col * Inches(6.2)
    y = Inches(1.8) + row * Inches(2.5)
    shape(sl, x, y, Inches(5.8), Inches(2.2), fill=CARD_WHITE)
    rect(sl, x, y, Pt(5), Inches(2.2), fill=RED)
    txt(sl, x + Inches(0.3), y + Inches(0.2), Inches(5.2), Inches(0.4),
        title, size=18, color=NAVY, bold=True)
    txt(sl, x + Inches(0.3), y + Inches(0.7), Inches(5.2), Inches(1.4),
        desc, size=13, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — The Solution (high level)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "The Solution: Ask a Question, Get a Trusted Answer", size=34, color=NAVY, bold=True)

# User journey flow
steps = [
    ("Ask", "Type your question\nin plain English", ACCENT),
    ("Route", "AI picks the best\ndata strategy", DEEP_BLUE),
    ("Compute", "Data + Docs retrieved\nMetrics compiled", GREEN),
    ("Validate", "5 automated checks\n+ AI quality review", AMBER),
    ("Answer", "Grounded answer\nwith confidence score", ACCENT),
]

for i, (label, desc, color) in enumerate(steps):
    x = Inches(0.5) + i * Inches(2.5)
    y = Inches(1.6)

    shape(sl, x, y, Inches(2.1), Inches(2.6), fill=CARD_WHITE)
    rect(sl, x, y, Inches(2.1), Pt(5), fill=color)

    s = sl.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.65), y + Inches(0.3), Inches(0.8), Inches(0.8))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    tf = s.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].text = str(i + 1)
    tf.paragraphs[0].font.size = Pt(24)
    tf.paragraphs[0].font.color.rgb = WHITE
    tf.paragraphs[0].font.bold = True

    txt(sl, x + Inches(0.15), y + Inches(1.2), Inches(1.8), Inches(0.4),
        label, size=18, color=color, bold=True, align=PP_ALIGN.CENTER)
    txt(sl, x + Inches(0.15), y + Inches(1.65), Inches(1.8), Inches(0.8),
        desc, size=12, color=MID_GRAY, align=PP_ALIGN.CENTER)

    if i < len(steps) - 1:
        arrow = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                     x + Inches(2.15), y + Inches(1.1), Inches(0.3), Inches(0.2))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = LIGHT_GRAY
        arrow.line.fill.background()

# Bottom value props
txt(sl, Inches(0.8), Inches(4.6), Inches(11), Inches(0.5),
    "What Makes This Different", size=20, color=NAVY, bold=True)

props = [
    ("Not a chatbot", "Every answer is backed by real data from your databases and actual policy documents — not LLM imagination"),
    ("Not a dashboard", "Handles questions that span multiple data sources, including questions that require chaining one source into another"),
    ("Not a report builder", "Answers come in seconds, not hours. Registered KPIs produce the exact same SQL every time, eliminating analyst variance"),
]

for i, (title, desc) in enumerate(props):
    x = Inches(0.5) + i * Inches(4.1)
    shape(sl, x, Inches(5.2), Inches(3.8), Inches(1.8), fill=BLUE_BG)
    txt(sl, x + Inches(0.2), Inches(5.3), Inches(3.4), Inches(0.4),
        title, size=15, color=ACCENT, bold=True)
    txt(sl, x + Inches(0.2), Inches(5.75), Inches(3.4), Inches(1.1),
        desc, size=12, color=DARK_TEXT)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — Business Impact / Metrics
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, NAVY)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "Business Impact", size=34, color=WHITE, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
    "From hours to seconds. From inconsistent to deterministic.", size=17, color=LIGHT_GRAY)

stat_card(sl, Inches(0.5), Inches(1.8), Inches(2.8), Inches(1.5),
          "< 5s", "Average query response", ACCENT, DEEP_BLUE)
stat_card(sl, Inches(3.6), Inches(1.8), Inches(2.8), Inches(1.5),
          "100%", "Reproducible metric SQL", GREEN, DEEP_BLUE)
stat_card(sl, Inches(6.7), Inches(1.8), Inches(2.8), Inches(1.5),
          "7", "Registered KPIs", AMBER, DEEP_BLUE)
stat_card(sl, Inches(9.8), Inches(1.8), Inches(2.8), Inches(1.5),
          "379", "Automated tests", ACCENT, DEEP_BLUE)

# Before/After comparison
shape(sl, Inches(0.5), Inches(3.8), Inches(5.8), Inches(3.2), fill=DEEP_BLUE)
txt(sl, Inches(0.7), Inches(3.9), Inches(5.4), Inches(0.5),
    "Before: Manual Process", size=20, color=RED, bold=True)
before = [
    "30-60 minutes per compliance question",
    "Analyst writes SQL, cross-references docs manually",
    "Different analysts = different answers",
    "Audit trail = email threads + spreadsheets",
    "Adding a new KPI = weeks of development",
]
bullets(sl, Inches(0.7), Inches(4.5), Inches(5.4), Inches(2.3),
        before, size=14, color=LIGHT_GRAY, spacing=Pt(6))

shape(sl, Inches(7.0), Inches(3.8), Inches(5.8), Inches(3.2), fill=DEEP_BLUE)
txt(sl, Inches(7.2), Inches(3.9), Inches(5.4), Inches(0.5),
    "After: AI Engine", size=20, color=GREEN, bold=True)
after = [
    "2-5 seconds per question, any complexity",
    "AI selects strategy, generates SQL, retrieves docs",
    "Registered metrics: same question = same SQL, always",
    "Full audit trail in PostgreSQL, queryable per session",
    "Adding a KPI = one YAML definition + one golden test",
]
bullets(sl, Inches(7.2), Inches(4.5), Inches(5.4), Inches(2.3),
        after, size=14, color=LIGHT_GRAY, spacing=Pt(6))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — What You Can Ask (Use Cases)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "What You Can Ask", size=34, color=NAVY, bold=True)

cases = [
    ("Metric Questions",
     "\"What is the compliance effectiveness score?\"\n"
     "\"Show me the department risk scores\"",
     "Deterministic SQL — same answer every time",
     GREEN, GREEN_BG),
    ("Data Queries",
     "\"How many critical violations in Q1 2024?\"\n"
     "\"Top 5 departments by number of violations\"",
     "AI generates and validates SQL against your schema",
     ACCENT, BLUE_BG),
    ("Policy Lookups",
     "\"What is our KYC policy for high-risk customers?\"\n"
     "\"Differences between AML and KYC policies\"",
     "Semantic search across all compliance documents",
     DEEP_BLUE, BLUE_BG),
    ("Hybrid Questions",
     "\"What is the compliance cost for 2024?\"\n"
     "(needs formula from docs + data from database)",
     "Chains documents into SQL automatically",
     AMBER, AMBER_BG),
    ("Trend Analysis",
     "\"How have violations changed over the last 4 quarters?\"\n"
     "\"Average resolution time for critical violations\"",
     "Time-series queries with metric thresholds",
     ACCENT, BLUE_BG),
    ("Follow-up Questions",
     "\"Break that down by department\"\n"
     "\"What does our policy say about those?\"",
     "Multi-turn conversations with context tracking",
     DEEP_BLUE, BLUE_BG),
]

for i, (title, examples, note, accent, bg) in enumerate(cases):
    col = i % 3
    row = i // 3
    x = Inches(0.3) + col * Inches(4.2)
    y = Inches(1.3) + row * Inches(3.0)

    shape(sl, x, y, Inches(3.9), Inches(2.7), fill=bg)
    rect(sl, x, y, Inches(3.9), Pt(4), fill=accent)
    txt(sl, x + Inches(0.2), y + Inches(0.15), Inches(3.5), Inches(0.35),
        title, size=16, color=accent, bold=True)
    txt(sl, x + Inches(0.2), y + Inches(0.55), Inches(3.5), Inches(1.3),
        examples, size=12, color=DARK_TEXT, font="Consolas")
    txt(sl, x + Inches(0.2), y + Inches(2.0), Inches(3.5), Inches(0.5),
        note, size=11, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — How It Works (Simple)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "How It Works", size=34, color=NAVY, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
    "The platform decides the best strategy for each question automatically", size=17, color=MID_GRAY)

# Simple flow
flow_items = [
    ("Your Question", "Plain English, any complexity", ACCENT, Inches(0.5)),
    ("Smart Router", "AI classifies into 1 of 5 strategies", DEEP_BLUE, Inches(2.8)),
    ("Data Retrieval", "SQL from Oracle + documents from vector store", GREEN, Inches(5.1)),
    ("Quality Check", "5 automated validators + AI reviewer", AMBER, Inches(7.4)),
    ("Trusted Answer", "With confidence score + full audit trail", ACCENT, Inches(9.7)),
]

for label, desc, color, x in flow_items:
    y = Inches(1.8)
    shape(sl, x, y, Inches(2.1), Inches(1.8), fill=CARD_WHITE)
    rect(sl, x, y, Inches(2.1), Pt(5), fill=color)
    txt(sl, x + Inches(0.15), y + Inches(0.2), Inches(1.8), Inches(0.4),
        label, size=15, color=color, bold=True, align=PP_ALIGN.CENTER)
    txt(sl, x + Inches(0.15), y + Inches(0.7), Inches(1.8), Inches(0.9),
        desc, size=12, color=MID_GRAY, align=PP_ALIGN.CENTER)

for x_start in [Inches(2.65), Inches(4.95), Inches(7.25), Inches(9.55)]:
    a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x_start, Inches(2.55), Inches(0.2), Inches(0.15))
    a.fill.solid()
    a.fill.fore_color.rgb = LIGHT_GRAY
    a.line.fill.background()

# Key insight boxes
txt(sl, Inches(0.8), Inches(4.0), Inches(11), Inches(0.5),
    "Key Innovation: Two Paths to SQL", size=20, color=NAVY, bold=True)

shape(sl, Inches(0.5), Inches(4.6), Inches(5.8), Inches(2.5), fill=GREEN_BG)
txt(sl, Inches(0.7), Inches(4.7), Inches(5.4), Inches(0.4),
    "Registered Metrics (7 KPIs)", size=18, color=GREEN, bold=True)
bullets(sl, Inches(0.7), Inches(5.2), Inches(5.4), Inches(1.8), [
    "Compliance Effectiveness Score, Audit Resolution Rate,",
    "Regulatory Exposure, Department Risk Score, and 3 more",
    "",
    "SQL generated from structured definitions — no AI interpretation",
    "Same question = same SQL, every time. Fully reproducible.",
], size=13, color=DARK_TEXT, spacing=Pt(3))

shape(sl, Inches(7.0), Inches(4.6), Inches(5.8), Inches(2.5), fill=BLUE_BG)
txt(sl, Inches(7.2), Inches(4.7), Inches(5.4), Inches(0.4),
    "Ad-hoc Questions (anything else)", size=18, color=ACCENT, bold=True)
bullets(sl, Inches(7.2), Inches(5.2), Inches(5.4), Inches(1.8), [
    "\"Top 5 departments by loss amount\"",
    "\"Which departments have both violations and audit findings?\"",
    "",
    "AI generates SQL with 3-layer validation + retry",
    "Column references checked before execution. Errors self-correct.",
], size=13, color=DARK_TEXT, spacing=Pt(3))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — Trust & Compliance
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "Built for Regulators", size=34, color=NAVY, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
    "Every answer can be fully explained: what data was used, how it was calculated, and why",
    size=17, color=MID_GRAY)

pillars = [
    ("Reproducibility",
     "Registered metrics produce identical SQL from identical questions. "
     "No analyst variance. No \"it depends on who you ask.\" "
     "Compile the same metric today and next year — same SQL.",
     GREEN),
    ("Full Audit Trail",
     "Every query is logged in PostgreSQL with: the question, routing decision, "
     "SQL executed, data returned, answer generated, review score, and confidence level. "
     "Queryable via /audit/{session_id}.",
     ACCENT),
    ("Confidence Scoring",
     "Every answer includes a trust signal: HIGH (deterministic metric, validated), "
     "MEDIUM (AI-generated SQL, good review), LOW (retries or validation issues). "
     "Users know exactly how much to trust each response.",
     AMBER),
    ("Data Protection",
     "PII (emails, SSNs, phone numbers, card numbers) is automatically detected and redacted "
     "before answers reach users. SQL injection is blocked at 3 layers. "
     "All queries are SELECT-only by design.",
     RED),
]

for i, (title, desc, color) in enumerate(pillars):
    col = i % 2
    row = i // 2
    x = Inches(0.5) + col * Inches(6.2)
    y = Inches(1.7) + row * Inches(2.6)
    shape(sl, x, y, Inches(5.8), Inches(2.3), fill=CARD_WHITE)
    rect(sl, x, y, Pt(5), Inches(2.3), fill=color)
    txt(sl, x + Inches(0.3), y + Inches(0.15), Inches(5.2), Inches(0.4),
        title, size=20, color=color, bold=True)
    txt(sl, x + Inches(0.3), y + Inches(0.65), Inches(5.2), Inches(1.5),
        desc, size=13, color=DARK_TEXT)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — Sample API Response
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, NAVY)

txt(sl, Inches(0.8), Inches(0.3), Inches(11), Inches(0.6),
    "What Users See", size=34, color=WHITE, bold=True)

# Question
shape(sl, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.8), fill=DEEP_BLUE)
txt(sl, Inches(0.7), Inches(1.25), Inches(1.5), Inches(0.5),
    "Question:", size=14, color=MID_GRAY)
txt(sl, Inches(2.2), Inches(1.25), Inches(10), Inches(0.6),
    "\"What is the compliance effectiveness score for the Legal department?\"",
    size=18, color=WHITE, bold=True, font="Calibri")

# Answer
shape(sl, Inches(0.5), Inches(2.3), Inches(7.5), Inches(3.0), fill=DEEP_BLUE)
txt(sl, Inches(0.7), Inches(2.4), Inches(7), Inches(0.4),
    "Answer", size=14, color=MID_GRAY)
txt(sl, Inches(0.7), Inches(2.8), Inches(7), Inches(2.3),
    "The Compliance Effectiveness Score for the Legal department is 55.0%, "
    "which is rated as Warning (60-80% threshold).\n\n"
    "This is based on the compliance_effectiveness_score metric (v1.0): "
    "55 out of 100 violations in Legal have been resolved. "
    "The department should prioritize closing open violations to improve this score above 80%.",
    size=15, color=WHITE)

# Metadata cards on the right
meta = [
    ("Route", "sql_only", ACCENT),
    ("Metric", "compliance_effectiveness\n_score v1.0", GREEN),
    ("Compiled", "Yes (deterministic)", GREEN),
    ("Confidence", "HIGH", GREEN),
    ("Review Score", "9.0 / 10", ACCENT),
]

for i, (label, value, color) in enumerate(meta):
    y = Inches(2.3) + i * Inches(0.6)
    shape(sl, Inches(8.3), y, Inches(4.5), Inches(0.55), fill=DEEP_BLUE)
    txt(sl, Inches(8.5), y + Inches(0.08), Inches(1.8), Inches(0.4),
        label, size=12, color=MID_GRAY)
    txt(sl, Inches(10.3), y + Inches(0.08), Inches(2.3), Inches(0.4),
        value, size=13, color=color, bold=True)

# Audit trail note
shape(sl, Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.4), fill=DEEP_BLUE)
txt(sl, Inches(0.7), Inches(5.7), Inches(11.5), Inches(0.4),
    "Audit Trail (available via /audit/{session_id})", size=14, color=AMBER, bold=True)
txt(sl, Inches(0.7), Inches(6.1), Inches(11.5), Inches(0.8),
    "Every step is recorded: cache check (miss) → context resolution → "
    "route decision (sql_only) → metric resolved (compliance_effectiveness_score@1.0) → "
    "SQL compiled deterministically → query executed → answer generated → "
    "reviewed (9.0/10) → confidence: HIGH → cached",
    size=13, color=LIGHT_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — Registered KPIs
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "Registered Business Metrics", size=34, color=NAVY, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
    "These KPIs produce deterministic SQL — no AI interpretation, fully reproducible",
    size=17, color=MID_GRAY)

kpis = [
    ("Compliance Effectiveness Score", "%",
     "What % of violations have been resolved?",
     "Closed violations / Total violations",
     "Warning: < 80%  |  Good: 80-95%  |  Excellent: > 95%"),
    ("Violation Severity Distribution", "count",
     "How many violations at each severity level?",
     "Count by Critical / High / Medium / Low",
     "Alert if any Critical + Open"),
    ("Audit Finding Resolution Rate", "%",
     "How effectively are audit findings addressed?",
     "Resolved findings / Total findings",
     "Critical: < 50%  |  Good: 75-90%"),
    ("Regulatory Exposure Index", "USD",
     "What is our financial exposure from open violations?",
     "Sum of financial impact, grouped by regulator",
     "Critical: > $1M  |  Warning: > $100K"),
    ("Department Risk Score", "score",
     "Which department carries the most risk?",
     "Weighted: violations + audit findings + risk events",
     "Critical: > 50  |  Warning: 20-50"),
    ("Control Coverage Ratio", "%",
     "What % of requirements have active controls?",
     "Requirements with controls / Total requirements",
     "Critical: < 70%  |  Good: 90%+"),
    ("Mean Time to Resolution", "days",
     "How fast do we close violations?",
     "Average days from violation to closure",
     "Critical: > 90 days  |  Excellent: < 15 days"),
]

# Header row
shape(sl, Inches(0.3), Inches(1.6), Inches(12.7), Inches(0.5), fill=ACCENT)
for col_x, col_text, col_w in [
    (Inches(0.4), "Metric", Inches(3.2)),
    (Inches(3.7), "Business Question", Inches(3.5)),
    (Inches(7.3), "Formula", Inches(2.8)),
    (Inches(10.2), "Thresholds", Inches(2.7)),
]:
    txt(sl, col_x, Inches(1.65), col_w, Inches(0.4),
        col_text, size=12, color=WHITE, bold=True)

for i, (name, unit, question, formula, thresholds) in enumerate(kpis):
    y = Inches(2.15) + i * Inches(0.68)
    bg = CARD_WHITE if i % 2 == 0 else SOFT_BG
    shape(sl, Inches(0.3), y, Inches(12.7), Inches(0.62), fill=bg)
    txt(sl, Inches(0.4), y + Inches(0.05), Inches(3.2), Inches(0.5),
        f"{name} ({unit})", size=12, color=NAVY, bold=True)
    txt(sl, Inches(3.7), y + Inches(0.05), Inches(3.5), Inches(0.5),
        question, size=11, color=DARK_TEXT)
    txt(sl, Inches(7.3), y + Inches(0.05), Inches(2.8), Inches(0.5),
        formula, size=11, color=MID_GRAY)
    txt(sl, Inches(10.2), y + Inches(0.05), Inches(2.7), Inches(0.5),
        thresholds, size=10, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 10 — Observability & Monitoring
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "Operational Visibility", size=34, color=NAVY, bold=True)
txt(sl, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
    "Real-time monitoring of every aspect of the platform", size=17, color=MID_GRAY)

dashboards = [
    ("Request Volume & Latency",
     "Track query volume, response times (p50/p95/p99), and error rates in real time. "
     "Alert when p95 exceeds 5 seconds.",
     ACCENT),
    ("Cache Performance",
     "Monitor cache hit rate to optimize cost. Target: 30%+ hit rate for recurring questions. "
     "Track cache TTL effectiveness.",
     GREEN),
    ("LLM Usage & Cost",
     "Token consumption by pipeline stage, cost per request, and per-model breakdown. "
     "Track Groq quota burn rate against daily limits.",
     AMBER),
    ("Answer Quality",
     "Review score distribution over time. Alert on sustained scores below 7.0. "
     "Track reflection loop frequency as a quality signal.",
     ACCENT),
    ("Route Distribution",
     "Which strategies are being used. Shifts in distribution may indicate changing question patterns "
     "or prompt drift in the router.",
     DEEP_BLUE),
    ("Circuit Breaker & Resilience",
     "LLM provider health, circuit breaker state, fallback activation rate. "
     "Alert when fallback model is serving more than 10% of requests.",
     RED),
]

for i, (title, desc, color) in enumerate(dashboards):
    col = i % 3
    row = i // 3
    x = Inches(0.3) + col * Inches(4.2)
    y = Inches(1.6) + row * Inches(2.7)
    icon_card(sl, x, y, Inches(3.9), Inches(2.4), title, desc, accent=color)


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 11 — Roadmap
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, OFF_WHITE)

txt(sl, Inches(0.8), Inches(0.4), Inches(11), Inches(0.6),
    "Roadmap", size=34, color=NAVY, bold=True)

phases = [
    ("Now", "Foundation",
     ["7 registered compliance KPIs",
      "5 routing strategies for hybrid queries",
      "Deterministic SQL compiler",
      "Full audit trail + confidence scoring",
      "379 automated tests + eval harness"],
     GREEN),
    ("Next", "Scale",
     ["Column-level pruning for wide tables (300+ cols)",
      "Real-data evaluation with known answers",
      "Per-request cost tracking in API response",
      "Additional database dialect support (Snowflake)",
      "Secrets manager integration (Vault)"],
     ACCENT),
    ("Future", "Enterprise",
     ["Multi-tenant deployment with RBAC",
      "Semantic layer integration (DataHub / OpenMetadata)",
      "Metric lineage and dependency tracking",
      "Self-service metric registration portal",
      "Natural language to dashboard generation"],
     DEEP_BLUE),
]

for i, (phase, title, items, color) in enumerate(phases):
    x = Inches(0.3) + i * Inches(4.2)
    y = Inches(1.3)

    shape(sl, x, y, Inches(3.9), Inches(5.7), fill=CARD_WHITE)
    rect(sl, x, y, Inches(3.9), Pt(5), fill=color)

    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             x + Inches(0.15), y + Inches(0.2), Inches(1.2), Inches(0.45))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    tf = s.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].text = phase
    tf.paragraphs[0].font.size = Pt(14)
    tf.paragraphs[0].font.color.rgb = WHITE
    tf.paragraphs[0].font.bold = True

    txt(sl, x + Inches(1.5), y + Inches(0.25), Inches(2.2), Inches(0.4),
        title, size=20, color=color, bold=True)

    bullets(sl, x + Inches(0.2), y + Inches(0.9), Inches(3.5), Inches(4.5),
            items, size=13, color=DARK_TEXT, spacing=Pt(10))


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 12 — Closing
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl, NAVY)

rect(sl, Inches(0), Inches(0), Inches(0.15), Inches(7.5), fill=ACCENT)

txt(sl, Inches(1.2), Inches(1.5), Inches(10), Inches(0.8),
    "ARIA — Analytical Risk Intelligence Agent", size=44, color=WHITE, bold=True)
txt(sl, Inches(1.2), Inches(2.5), Inches(10), Inches(0.6),
    "From questions to trusted, auditable answers in seconds", size=22, color=ACCENT)

closing_points = [
    "Ask any compliance or risk question in plain English",
    "Registered KPIs computed deterministically — no AI guessing",
    "Every answer validated by 5 automated checks + AI reviewer",
    "Full audit trail for every response — regulator-ready",
    "Confidence score on every answer — users know what to trust",
]
bullets(sl, Inches(1.2), Inches(3.5), Inches(10), Inches(2.5),
        closing_points, size=17, color=LIGHT_GRAY, spacing=Pt(14))

txt(sl, Inches(1.2), Inches(6.2), Inches(10), Inches(0.8),
    "Pranab Akhoury\npranab.akhoury@gmail.com",
    size=15, color=MID_GRAY, align=PP_ALIGN.LEFT)


# ── Save ──────────────────────────────────────────────────────────────
output = "Intelligence_AI_Engine_Business_Demo.pptx"
prs.save(output)
print(f"Presentation saved: {output}")
print(f"  Slides: {len(prs.slides)}")
