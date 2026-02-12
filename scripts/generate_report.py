"""
Generate a progress report PDF for Project Genesis.
For consultation with Deep Think and Sonnet.
"""

import os
import json
import datetime

# Try reportlab first, fall back to FPDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    USE_REPORTLAB = True
except ImportError:
    USE_REPORTLAB = False

OUTPUT_DIR = r"C:\Users\kyjan\研究\daily_report"
FIGURES_DIR = r"C:\Users\kyjan\研究\snn-genesis\figures"
RESULTS_DIR = r"C:\Users\kyjan\研究\snn-genesis\results"


def build_pdf_reportlab():
    """Build report with ReportLab."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    pdf_path = os.path.join(OUTPUT_DIR, f"genesis_progress_{today}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=20, spaceAfter=6, textColor=HexColor('#1a1a2e'),
    )
    h1_style = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontSize=15, spaceAfter=8, spaceBefore=14,
        textColor=HexColor('#16213e'),
    )
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=12, spaceAfter=6, spaceBefore=10,
        textColor=HexColor('#0f3460'),
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=10, spaceAfter=4, leading=14,
    )
    small_style = ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontSize=8, textColor=HexColor('#666666'),
    )
    code_style = ParagraphStyle(
        'Code', parent=styles['Normal'],
        fontName='Courier', fontSize=9, spaceAfter=4,
        leftIndent=10, backColor=HexColor('#f5f5f5'),
    )
    finding_style = ParagraphStyle(
        'Finding', parent=styles['Normal'],
        fontSize=10, spaceAfter=4, leading=14,
        leftIndent=15, borderColor=HexColor('#2ecc71'),
        borderWidth=2, borderPadding=5,
    )

    story = []

    # ── Title ──
    story.append(Paragraph(
        "Project Genesis: Progress Report", title_style))
    story.append(Paragraph(
        "Self-Evolving AI via SNN Chaotic Randomness", h2_style))
    story.append(Paragraph(
        f"Date: {today} &nbsp; | &nbsp; Author: Hiroto Funasaki &nbsp; | &nbsp; "
        f"Status: Phase 5 Complete", small_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
    story.append(Spacer(1, 6))

    # ── Executive Summary ──
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "Project Genesis unifies four independent SNN research papers into a "
        "self-evolving AI pipeline. The core hypothesis is that SNN chaotic dynamics "
        "(\"Edge of Chaos\") can generate higher-quality training data than Gaussian noise, "
        "enabling LLMs to self-improve through autonomous evolution loops.", body_style))
    story.append(Paragraph(
        "<b>Key Discovery:</b> SNN chaotic noise produces stable, monotonic self-evolution "
        "(20% → 10% → 0% nightmare resistance) while Gaussian noise oscillates "
        "(0% → 20% → 0%). This validates the \"Data Alchemy\" hypothesis.", body_style))
    story.append(Spacer(1, 4))

    # ── The 4 Papers ──
    story.append(Paragraph("Foundation: 4 Papers Unified", h1_style))

    papers_data = [
        ['Paper', 'Core Technology', 'Role in Genesis'],
        ['SNN-Comprypto v5', 'Chaotic reservoir, NIST-grade randomness', 'Noise source'],
        ['Hybrid SNN-LM v4', 'BitNet b1.58, spike+membrane readout', 'Efficient inference'],
        ['Brain vs Neumann v3', '11D hypercube, burst coding', 'Theoretical backbone'],
        ['AI Immune System v11', 'Canary head, Dream Catcher, Morpheus', 'Detection + training'],
    ]
    t = Table(papers_data, colWidths=[110, 210, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f9f9f9'), HexColor('#ffffff')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Grand Unification Hypothesis:</b> SNN's three capabilities "
        "(conversion, chaos, detection) all emerge from the same principle: "
        "\"Edge of Chaos\" dynamics.", body_style))

    # ── Experimental Results ──
    story.append(Paragraph("Experimental Results (Phases 1-5)", h1_style))

    # Phase 1
    story.append(Paragraph("Phase 1: SNN Randomness Validation", h2_style))
    p1_data = [
        ['Source', 'Prediction Rate', 'Chi-squared', 'Autocorrelation'],
        ['SNN', '1.54%', '228', '0.009'],
        ['numpy', '0.39%', '270', '0.008'],
        ['ANN', '100%', '25M', '0.316'],
    ]
    t = Table(p1_data, colWidths=[80, 100, 100, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f0fff0'), HexColor('#ffffff')]),
    ]))
    story.append(t)
    story.append(Paragraph("Result: SNN is 64.9x more random than ANN.", body_style))

    # Phase 2-4 summary
    story.append(Paragraph("Phases 2-4: Noise Injection → Data Gen → Vaccination", h2_style))
    p24_data = [
        ['Phase', 'Result', 'Metric'],
        ['P2: Noise Injection', 'SNN/torch ratio = 0.99x', 'Equivalent effectiveness'],
        ['P3: Dream Catcher v2', '150 samples, 98% heal rate', 'Nightmare diversity 42%'],
        ['P4: QLoRA Vaccination', 'Nightmare +6.7%', 'Clean accuracy preserved'],
    ]
    t = Table(p24_data, colWidths=[120, 160, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f0f8ff'), HexColor('#ffffff')]),
    ]))
    story.append(t)

    # Phase 5 — Key result
    story.append(Paragraph("Phase 5: Evolution Loop (Key Discovery)", h2_style))
    story.append(Paragraph(
        "3-round self-evolution comparing SNN Chaos (Genesis) vs torch.randn (Morpheus):",
        body_style))

    p5_data = [
        ['Round', 'Genesis (SNN)', 'Morpheus (randn)', 'Genesis Loss', 'Morpheus Loss'],
        ['0 (baseline)', '20%', '0%', '—', '—'],
        ['1', '10% ↓', '20% ↑ spike!', '1.33', '1.33'],
        ['2', '0% ✓', '0% ✓', '0.76', '0.75'],
        ['3', '0% ✓', '0% ✓', '0.43', '0.36'],
    ]
    t = Table(p5_data, colWidths=[75, 85, 95, 85, 85])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff5f5'), HexColor('#ffffff')]),
        ('BACKGROUND', (1, 4), (1, 4), HexColor('#d4edda')),  # Genesis final green
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Phase 5 chart
    chart_path = os.path.join(FIGURES_DIR, "phase5_evolution_loop.png")
    if os.path.exists(chart_path):
        img = Image(chart_path, width=170*mm, height=120*mm)
        story.append(img)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Key Findings:</b>", body_style))
    findings = [
        "Genesis (SNN): 20% → 10% → 0% — monotonic, stable evolution",
        "Morpheus (randn): 0% → 20% → 0% — unstable spike at Round 1",
        "Genesis maintains 100% clean accuracy; Morpheus dips to 90%",
        "SNN chaotic noise produces more stable self-evolution trajectories",
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", body_style))

    # ── New Discovery: 11D Hypercube ──
    story.append(PageBreak())
    story.append(Paragraph("New Discovery: 11D Hypercube → Canary Zone", h1_style))
    story.append(Paragraph(
        "Cross-paper analysis revealed that the 11D hypercube topology from "
        "\"Brain vs Neumann\" (Paper 3) may explain why canary heads cluster at "
        "30-55% model depth (the Universal Safety Zone from Paper 4).", body_style))
    story.append(Paragraph(
        "<b>Experiment:</b> Mapped Mistral-7B's 32 layers to 11D hypercube coordinates "
        "and computed hub layers (minimal average Hamming distance to all others).",
        body_style))
    story.append(Paragraph(
        "<b>Result:</b> Hub layers fall in the 30-55% zone, confirming the theory.",
        body_style))
    story.append(Paragraph(
        "This suggests that the brain's topological structure (11D, not 3D) naturally "
        "creates information processing hubs at specific depth ratios, and transformer "
        "models inadvertently replicate this pattern.", body_style))

    # ── Oracle LM Architecture ──
    story.append(Paragraph("Prototype: Oracle LM (SNN-ANN Hybrid)", h1_style))
    story.append(Paragraph(
        "Implemented a proof-of-concept SNN co-processor that sits at the canary layer "
        "of an ANN backbone (Mistral-7B). Three operational modes:", body_style))

    oracle_data = [
        ['Mode', 'Function', 'Status'],
        ['Monitor', 'Detect anomalies via SNN canary', 'Working ✓'],
        ['Evolve', 'Inject SNN chaos for data generation', 'Working ✓'],
        ['Heal', 'Self-repair detected anomalies', 'Working ✓'],
    ]
    t = Table(oracle_data, colWidths=[80, 220, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ]))
    story.append(t)
    story.append(Paragraph(
        "BitNet ternary weights achieve 42.5% sparsity (only {-1, 0, +1} values).",
        body_style))

    # ── Questions for Consultation ──
    story.append(Paragraph("Questions for Deep Think / Sonnet", h1_style))
    story.append(Paragraph(
        "The following questions need expert input to determine next steps:", body_style))
    questions = [
        "<b>Q1: Paper Readiness.</b> Are these findings sufficient for a standalone paper "
        "(\"Project Genesis: Self-Evolving AI via SNN Chaotic Data Alchemy\"), or do we "
        "need more experiments? Current weakness: only 3 rounds × 20 questions.",
        "<b>Q2: Statistical Rigor.</b> What additional experiments would strengthen the "
        "claim? More rounds? Larger question sets? Multiple model architectures? "
        "Cross-validation?",
        "<b>Q3: 11D Hypercube.</b> The hub-layer ↔ canary-zone correlation is suggestive. "
        "Is this a coincidence or a fundamental architectural principle? What would "
        "constitute a rigorous test?",
        "<b>Q4: Oracle LM Direction.</b> Should the SNN co-processor be: "
        "(a) a runtime safety module, (b) a training-time data generator, "
        "or (c) both? Which path has more novelty?",
        "<b>Q5: Integration with Existing Papers.</b> Should Genesis be a standalone paper, "
        "an extension of v11 (AI Immune System), or something else?",
    ]
    for q in questions:
        story.append(Paragraph(f"• {q}", body_style))
        story.append(Spacer(1, 3))

    # ── GitHub ──
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
    story.append(Paragraph(
        "Repository: github.com/hafufu-stack/snn-genesis", small_style))
    story.append(Paragraph(
        "All experiments reproducible. GPU: NVIDIA RTX 5080 Laptop (17.1GB VRAM).",
        small_style))

    doc.build(story)
    return pdf_path


def build_pdf_fallback():
    """Fallback: Generate a simple text-based PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        # Final fallback: just generate a markdown file
        return build_markdown_fallback()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    pdf_path = os.path.join(OUTPUT_DIR, f"genesis_progress_{today}.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Project Genesis: Progress Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Date: {today} | Author: Hiroto Funasaki", ln=True, align="C")
    pdf.ln(5)

    sections = [
        ("Executive Summary",
         "Project Genesis unifies 4 SNN papers into a self-evolving AI pipeline. "
         "Key finding: SNN chaos produces stable evolution (20->10->0%) while "
         "Gaussian noise oscillates (0->20->0%)."),
        ("Phase 1-4 Summary",
         "P1: SNN 64.9x more random than ANN. "
         "P2: SNN/torch ratio 0.99x. "
         "P3: 150 vaccine samples, 98% heal. "
         "P4: +6.7% nightmare resistance."),
        ("Phase 5: Evolution Loop (Key Discovery)",
         "Genesis (SNN): 20% -> 10% -> 0% (monotonic, stable)\n"
         "Morpheus (randn): 0% -> 20% -> 0% (unstable spike)\n"
         "Genesis loss: 1.33 -> 0.76 -> 0.43\n"
         "Genesis clean accuracy: 100% throughout"),
        ("New: 11D Hypercube -> Canary Zone",
         "Hub layers of 11D hypercube mapped onto Mistral-7B fall in "
         "30-55% Universal Safety Zone. Confirms theory from Brain vs Neumann."),
        ("Prototype: Oracle LM",
         "SNN-ANN hybrid co-processor with 3 modes: Monitor, Evolve, Heal. "
         "BitNet ternary weights achieve 42.5% sparsity."),
        ("Questions for Consultation",
         "Q1: Enough for a paper?\n"
         "Q2: What additional experiments?\n"
         "Q3: Is 11D hypercube correlation real?\n"
         "Q4: Oracle LM direction (safety/training/both)?\n"
         "Q5: Standalone paper or extension of v11?"),
    ]

    for title, text in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        for line in text.split("\n"):
            pdf.multi_cell(0, 6, line)
        pdf.ln(3)

    # Add chart if possible
    chart_path = os.path.join(FIGURES_DIR, "phase5_evolution_loop.png")
    if os.path.exists(chart_path):
        try:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Phase 5 Evolution Loop Chart", ln=True)
            pdf.image(chart_path, x=10, w=190)
        except Exception:
            pass

    pdf.output(pdf_path)
    return pdf_path


def build_markdown_fallback():
    """Last resort: markdown file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    md_path = os.path.join(OUTPUT_DIR, f"genesis_progress_{today}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Project Genesis: Progress Report\n")
        f.write(f"Date: {today}\n\n")
        f.write("(See snn-genesis README.md for full details)\n")
    return md_path


if __name__ == "__main__":
    if USE_REPORTLAB:
        path = build_pdf_reportlab()
        print(f"PDF generated (ReportLab): {path}")
    else:
        print("ReportLab not found, trying fpdf2...")
        path = build_pdf_fallback()
        print(f"PDF generated: {path}")
