"""
Generate evening update progress report PDF.
Includes Scale-Up results + Catastrophic Forgetting analysis.
"""

import os
import json
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)

OUTPUT_DIR = r"C:\Users\kyjan\研究\daily_report"
FIGURES_DIR = r"C:\Users\kyjan\研究\snn-genesis\figures"
RESULTS_DIR = r"C:\Users\kyjan\研究\snn-genesis\results"


def build_pdf():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%H%M")
    pdf_path = os.path.join(OUTPUT_DIR, f"genesis_progress_{today}_{now}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=18, spaceAfter=4, textColor=HexColor('#1a1a2e'))
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=14, spaceAfter=6, spaceBefore=12, textColor=HexColor('#16213e'))
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, spaceAfter=4, spaceBefore=8, textColor=HexColor('#0f3460'))
    body = ParagraphStyle('B', parent=styles['Normal'], fontSize=9.5, spaceAfter=3, leading=13)
    small = ParagraphStyle('S', parent=styles['Normal'], fontSize=8, textColor=HexColor('#666666'))
    alert = ParagraphStyle('A', parent=styles['Normal'], fontSize=9.5, spaceAfter=3, leading=13,
                           backColor=HexColor('#fff3cd'), borderColor=HexColor('#ffc107'),
                           borderWidth=1, borderPadding=6, leftIndent=10)
    critical = ParagraphStyle('C', parent=styles['Normal'], fontSize=9.5, spaceAfter=3, leading=13,
                              backColor=HexColor('#f8d7da'), borderColor=HexColor('#dc3545'),
                              borderWidth=1, borderPadding=6, leftIndent=10)

    story = []

    # ── Title ──
    story.append(Paragraph("Project Genesis: Progress Report (Evening Update)", title_style))
    story.append(Paragraph(
        f"Date: {today} {now[:2]}:{now[2:]} &nbsp;|&nbsp; Author: Hiroto Funasaki &nbsp;|&nbsp; "
        f"Status: Scale-Up Complete", small))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
    story.append(Spacer(1, 4))

    # ── Executive Summary ──
    story.append(Paragraph("Executive Summary", h1))
    story.append(Paragraph(
        "4-paper unified SNN self-evolution pipeline. Phase 5 original (20Q x 3R) showed "
        "stable Genesis (SNN) convergence. <b>Scale-Up (100Q x 5R) revealed Catastrophic "
        "Forgetting in both branches</b> — training pipeline needs redesign.", body))

    # ── Original Phase 5 Success ──
    story.append(Paragraph("Phase 5 Original (Success)", h1))
    p5_data = [
        ['Round', 'Genesis NM', 'Morpheus NM', 'Genesis Clean', 'Morpheus Clean'],
        ['0', '20%', '0%', '100%', '100%'],
        ['1', '10% ↓', '20% ↑ spike', '100%', '90% ↓'],
        ['2', '0% ✓', '0% ✓', '100%', '100%'],
        ['3', '0% ✓', '0% ✓', '100%', '100%'],
    ]
    t = Table(p5_data, colWidths=[55, 85, 85, 85, 85])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f0fff0'), HexColor('#ffffff')]),
    ]))
    story.append(t)
    story.append(Paragraph(
        "<b>Key Finding:</b> Genesis shows monotonic improvement (20→10→0%) while "
        "Morpheus oscillates (0→20→0%). Config: 20 questions, 3 rounds, Loss 1.33→0.43.", body))

    # Phase 5 chart
    chart = os.path.join(FIGURES_DIR, "phase5_evolution_loop.png")
    if os.path.exists(chart):
        story.append(Image(chart, width=160*mm, height=110*mm))

    # ── Scale-Up Results (NEW) ──
    story.append(PageBreak())
    story.append(Paragraph("⚠ Scale-Up Experiment (Catastrophic Forgetting)", h1))
    story.append(Paragraph(
        "<b>Config:</b> 100 questions (60 clean + 40 nightmare), 5 rounds, "
        "Mistral-7B-Instruct-v0.3, QLoRA r=8, σ=0.10, transformers 5.0.0", body))

    su_data = [
        ['Round', 'Genesis Clean', 'Genesis NM', 'Morpheus Clean', 'Morpheus NM', 'G Loss', 'M Loss'],
        ['0 (base)', '70%', '70%', '70%', '70%', '—', '—'],
        ['1', '10% ↓↓', '100% ↑↑', '10% ↓↓', '100% ↑↑', '4.35', '4.49'],
        ['2', '10%', '100%', '10%', '100%', '3.62', '3.48'],
        ['3', '10%', '100%', '10%', '100%', '3.46', '3.47'],
        ['4', '10%', '100%', '10%', '100%', '3.59', '3.41'],
        ['5', '10%', '100%', '10%', '100%', '3.53', '3.47'],
    ]
    t = Table(su_data, colWidths=[55, 65, 60, 70, 65, 50, 50])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff5f5'), HexColor('#ffffff')]),
        ('BACKGROUND', (1, 2), (4, -1), HexColor('#fce4e4')),  # Highlight bad cells
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Scale-up chart
    su_chart = os.path.join(FIGURES_DIR, "phase5_scaleup.png")
    if os.path.exists(su_chart):
        story.append(Image(su_chart, width=170*mm, height=55*mm))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>CRITICAL:</b> Both Genesis and Morpheus collapsed at Round 1. "
        "Clean accuracy dropped 70%→10%, Nightmare rose 70%→100%. "
        "No difference between SNN and Gaussian noise — the training pipeline "
        "itself is broken.", critical))

    # ── Root Cause Analysis ──
    story.append(Paragraph("Root Cause Analysis", h2))
    causes = [
        ['Factor', 'Phase 5 Original', 'Phase 5 Scale-Up', 'Impact'],
        ['Training Loss', '1.33 (healthy)', '4.35 (very high)', 'Model not learning correctly'],
        ['Data Format', 'Phase 5 internal', '<s>[INST]...[/INST]...</s>', 'Possible double-tokenization'],
        ['transformers', 'older version', '5.0.0 (breaking)', 'API changes in SFTTrainer'],
        ['trl', 'older version', 'latest (no max_seq_length)', 'Config parameter removed'],
        ['Questions', '20 (manual)', '100 (programmatic)', 'More diverse but same pipeline'],
    ]
    t = Table(causes, colWidths=[70, 90, 90, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ff9800')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff8e1'), HexColor('#ffffff')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Most Likely Cause:</b> The healed nightmare data + clean Q&amp;A were formatted "
        "as raw text strings, but SFTTrainer in trl latest may be applying its own chat "
        "template on top, resulting in double-wrapping. Loss=4.35 confirms the model is "
        "essentially seeing corrupted tokens.", alert))

    # ── 11D Topology Results ──
    story.append(Paragraph("11D Hypercube Topology Verification", h1))
    story.append(Paragraph(
        "Tested 10 model architectures. <b>All predicted hub centers fall in 0-55%</b> "
        "(Universal Zone), but precise canary prediction failed (0/4 exact match).", body))

    topo_data = [
        ['Model', 'Layers', 'Predicted Center', 'Known Canary', 'Match?'],
        ['Mistral-7B', '32', '8.6%', '31.3% (L10)', 'No'],
        ['Llama-3.2-3B', '28', '17.9%', '46.4% (L13)', 'No'],
        ['Phi-2', '32', '8.6%', '31.3%', 'No'],
        ['Llama-3.1-8B', '32', '8.6%', '31.3%', 'No'],
        ['GPT-2', '12', '50.0%', '?', 'Prediction'],
        ['Qwen2.5-14B', '48', '25.0%', '?', 'Prediction'],
        ['Llama-3.1-70B', '80', '36.2%', '?', 'Prediction'],
    ]
    t = Table(topo_data, colWidths=[85, 45, 80, 80, 50])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ]))
    story.append(t)
    story.append(Spacer(1, 3))

    topo_chart = os.path.join(FIGURES_DIR, "topology_verification.png")
    if os.path.exists(topo_chart):
        story.append(Image(topo_chart, width=170*mm, height=75*mm))

    story.append(Paragraph(
        "<b>Assessment:</b> Linear layer→11bit mapping is too simplistic. The 11D topology "
        "idea has merit (all centers in 0-55%) but needs a better mapping algorithm "
        "(possibly based on actual attention flow, not just layer index).", alert))

    # ── Questions for Consultation ──
    story.append(PageBreak())
    story.append(Paragraph("Updated Questions for Deep Think / Sonnet", h1))
    questions = [
        "<b>Q1: Training Pipeline Fix.</b> The QLoRA scale-up produced Loss=4.35 (vs 1.33 "
        "in original). Most likely cause: data format incompatibility with trl latest. "
        "Should we (a) downgrade trl, (b) fix data format, or (c) switch to manual "
        "training loop without SFTTrainer?",
        "<b>Q2: Evaluation Robustness.</b> Baseline Clean=70% (10 questions) is fragile. "
        "Should we use a standardized benchmark (MMLU, TruthfulQA) instead of custom Q&amp;A?",
        "<b>Q3: 11D Mapping Algorithm.</b> Linear mapping failed. Alternative approaches: "
        "(a) map based on attention entropy profile, (b) use actual gradient flow as "
        "topology, (c) fit hypercube dimensions to match empirical canary data?",
        "<b>Q4: Catastrophic Forgetting Prevention.</b> Even the original Phase 5 showed "
        "some forgetting (Morpheus Clean 100→90%). Proposed solutions: "
        "(a) Elastic Weight Consolidation, (b) replay buffer of clean data, "
        "(c) lower learning rate + more clean data in mix?",
        "<b>Q5: Paper Strategy.</b> Given the scale-up failure, should we: "
        "(a) fix pipeline and rerun before writing, (b) include the failure as a "
        "'lessons learned' section, or (c) focus on the original 20Q results + "
        "theoretical contributions?",
    ]
    for q in questions:
        story.append(Paragraph(f"• {q}", body))
        story.append(Spacer(1, 2))

    # ── Timeline ──
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
    story.append(Paragraph(
        "Repository: github.com/hafufu-stack/snn-genesis (commit ca99b07)", small))
    story.append(Paragraph(
        "GPU: NVIDIA RTX 5080 Laptop (17.1GB). Scale-up ran 90min (19:54-21:25).", small))

    doc.build(story)
    return pdf_path


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF generated: {path}")
