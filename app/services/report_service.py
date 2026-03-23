"""
app/services/report_service.py
Generates professional PDF and HTML reports using ReportLab.
No external API references — all content from model pipeline output.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)


def _build_pdf(output_path: str, d: dict):
    """Build PDF report using ReportLab platypus layout engine."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    # ── Colour palette ──────────────────────────────────────────────────────
    NAVY   = colors.HexColor("#0f172a")
    INDIGO = colors.HexColor("#4f46e5")
    VIOLET = colors.HexColor("#7c3aed")
    SLATE  = colors.HexColor("#64748b")
    LGRAY  = colors.HexColor("#f1f5f9")
    MGRAY  = colors.HexColor("#e2e8f0")
    RED    = colors.HexColor("#dc2626")
    AMBER  = colors.HexColor("#d97706")
    GREEN  = colors.HexColor("#16a34a")
    WHITE  = colors.white
    BLACK  = colors.HexColor("#0f172a")

    risk_color = {"LOW": GREEN, "MODERATE": AMBER, "HIGH": RED}.get(
        d.get("risk_level", ""), INDIGO
    )

    # ── Page layout ─────────────────────────────────────────────────────────
    # Explicit margins so header and content never overlap
    LEFT_MARGIN  = 20 * mm
    RIGHT_MARGIN = 20 * mm
    TOP_MARGIN   = 22 * mm
    BOT_MARGIN   = 22 * mm
    W, H  = A4
    CW    = W - LEFT_MARGIN - RIGHT_MARGIN   # usable content width

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOT_MARGIN,
    )

    # ── Paragraph styles ─────────────────────────────────────────────────────
    def PS(name, **kw):
        return ParagraphStyle(name, **kw)

    # All styles use explicit fontName/fontSize so they never inherit
    # a different default that would cause size/overflow issues
    org_s   = PS("Org",  fontName="Helvetica-Bold",  fontSize=18,
                 textColor=NAVY, spaceAfter=0, leading=22)
    title_s = PS("Tit",  fontName="Helvetica-Bold",  fontSize=13,
                 textColor=NAVY, spaceBefore=6, spaceAfter=2, leading=16)
    sub_s   = PS("Sub",  fontName="Helvetica",        fontSize=8,
                 textColor=SLATE, spaceAfter=3, leading=11)
    meta_s  = PS("Meta", fontName="Courier",          fontSize=7,
                 textColor=SLATE, alignment=TA_RIGHT, leading=10)
    sec_s   = PS("Sec",  fontName="Helvetica-Bold",   fontSize=8,
                 textColor=INDIGO, spaceAfter=4, spaceBefore=10, leading=11)
    body_s  = PS("Bod",  fontName="Helvetica",         fontSize=9,
                 textColor=NAVY, spaceAfter=4, leading=13)
    mono_s  = PS("Mono", fontName="Courier",           fontSize=7.5,
                 textColor=INDIGO, spaceAfter=3, leading=12)
    lbl_s   = PS("Lbl",  fontName="Helvetica-Bold",   fontSize=8,
                 textColor=SLATE, leading=11)
    tag_s   = PS("Tag",  fontName="Helvetica-Bold",   fontSize=8,
                 textColor=VIOLET, leading=11)
    red_s   = PS("Red",  fontName="Helvetica",         fontSize=8,
                 textColor=RED, leading=12)
    foot_s  = PS("Ft",   fontName="Helvetica",         fontSize=7.5,
                 textColor=SLATE, leading=10)
    ftr_s   = PS("Ftr",  fontName="Helvetica",         fontSize=7.5,
                 textColor=SLATE, alignment=TA_RIGHT, leading=10)

    # ── Helper: section heading ──────────────────────────────────────────────
    def section(title):
        story.append(Paragraph(title.upper(), sec_s))
        story.append(HRFlowable(
            width="100%", thickness=0.4,
            color=colors.HexColor("#cbd5e1"), spaceAfter=4
        ))

    # ── Build story ──────────────────────────────────────────────────────────
    story = []

    # ── HEADER BLOCK ──────────────────────────────────────────────────────────
    # Two-column table: left = org name + title, right = report metadata
    # Fixed column widths prevent text overlap
    hdr_left = [
        Paragraph("NeuroScan AI", org_s),
        Paragraph("Neurological Risk Assessment Report", title_s),
        Paragraph(
            "Multimodal Handwriting &amp; Speech Analysis · NSS Classification",
            sub_s
        ),
    ]
    hdr_right = [
        Paragraph(
            f"Report ID: {d.get('report_id','—')}<br/>"
            f"Date: {d.get('timestamp','')[:10]}<br/>"
            f"Time: {d.get('timestamp','')[ 11:19] if 'T' in d.get('timestamp','') else ''}<br/>"
            f"Platform: v{settings.APP_VERSION}",
            meta_s,
        ),
    ]

    hdr_tbl = Table(
        [[hdr_left, hdr_right]],
        colWidths=[CW * 0.60, CW * 0.40],
    )
    hdr_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(hdr_tbl)
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=10))

    # ── PATIENT INFO ───────────────────────────────────────────────────────────
    ts  = d.get("timestamp", "")
    pi  = Table(
        [[
            Paragraph(f"<b>Patient:</b> {d.get('patient_name') or 'Anonymous'}", body_s),
            Paragraph(f"<b>Date:</b> {ts[:10]}", body_s),
            Paragraph(f"<b>Report ID:</b> {d.get('report_id','—')}", body_s),
        ]],
        colWidths=[CW / 3] * 3,
    )
    pi.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(pi)
    story.append(Spacer(1, 10))

    # ── NSS SCORE PANEL ────────────────────────────────────────────────────────
    nss_val  = d.get("nss_score",  0.0)
    z_val    = d.get("z_score",    0.0)
    conf_val = d.get("confidence", 0.0)
    emoji    = d.get("risk_emoji", "")
    rlvl     = d.get("risk_level", "—")

    score_tbl = Table(
        [
            [
                Paragraph("NSS SCORE",  PS("h1",fontName="Helvetica",fontSize=7,textColor=WHITE,alignment=TA_CENTER,leading=10)),
                Paragraph("RISK LEVEL", PS("h2",fontName="Helvetica",fontSize=7,textColor=WHITE,alignment=TA_CENTER,leading=10)),
                Paragraph("Z-SCORE",    PS("h3",fontName="Helvetica",fontSize=7,textColor=WHITE,alignment=TA_CENTER,leading=10)),
                Paragraph("CONFIDENCE", PS("h4",fontName="Helvetica",fontSize=7,textColor=WHITE,alignment=TA_CENTER,leading=10)),
            ],
            [
                Paragraph(f"<b>{nss_val:.4f}</b>",
                    PS("sv",fontName="Courier-Bold",fontSize=20,textColor=risk_color,alignment=TA_CENTER,leading=24)),
                Paragraph(f"<b>{emoji} {rlvl}</b>",
                    PS("rv",fontName="Helvetica-Bold",fontSize=12,textColor=risk_color,alignment=TA_CENTER,leading=16)),
                Paragraph(f"<b>{z_val:.5f}</b>",
                    PS("zv",fontName="Courier-Bold",fontSize=11,textColor=INDIGO,alignment=TA_CENTER,leading=14)),
                Paragraph(f"<b>{conf_val*100:.1f}%</b>",
                    PS("cv",fontName="Courier-Bold",fontSize=11,textColor=VIOLET,alignment=TA_CENTER,leading=14)),
            ],
        ],
        colWidths=[CW / 4] * 4,
    )
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 10))

    # ── FORMULA ────────────────────────────────────────────────────────────────
    section("Computation Pipeline")
    formula_text = d.get("formula_display", "").replace("\n", "<br/>")
    story.append(Paragraph(formula_text, mono_s))
    story.append(Spacer(1, 6))

    # ── METRICS ────────────────────────────────────────────────────────────────
    def metrics_table(title, metrics):
        section(title)
        items = list(metrics.items())
        rows  = []
        for i in range(0, len(items), 2):
            row = []
            for k, v in items[i:i+2]:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                row += [
                    Paragraph(k.replace("_", " ").title(), lbl_s),
                    Paragraph(val_str, mono_s),
                ]
            while len(row) < 4:
                row += [Paragraph("", lbl_s), Paragraph("", mono_s)]
            rows.append(row)
        tbl = Table(
            rows,
            colWidths=[CW * 0.28, CW * 0.22, CW * 0.28, CW * 0.22],
        )
        tbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LGRAY]),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

    metrics_table("Handwriting Stroke Metrics", d.get("stroke_metrics", {}))
    metrics_table("Acoustic Speech Metrics",    d.get("acoustic_metrics", {}))

    # ── CLINICAL ANALYSIS ──────────────────────────────────────────────────────
    ai = d.get("ai_analysis", {})

    def ai_para(heading, text):
        if not text:
            return
        section(heading)
        story.append(Paragraph(str(text), body_s))
        story.append(Spacer(1, 4))

    def ai_list_section(heading, items):
        if not items:
            return
        section(heading)
        for item in (items or []):
            story.append(Paragraph(f"&#9658;  {item}", body_s))
        story.append(Spacer(1, 4))

    ai_para("Clinical Summary",       ai.get("clinical_summary", ""))
    ai_para("Handwriting Findings",   ai.get("handwriting_findings", ""))
    ai_para("Speech Findings",        ai.get("speech_findings", ""))

    ai_list_section("Neurological Indicators", ai.get("neurological_indicators", []))

    # Differential diagnosis as tags
    diff = ai.get("differential_diagnosis", [])
    if diff:
        section("Differential Diagnosis")
        tag_text = "    |    ".join(f"[ {item} ]" for item in diff)
        story.append(Paragraph(tag_text, tag_s))
        story.append(Spacer(1, 6))

    ai_list_section("Clinical Recommendations", ai.get("recommendations", []))
    ai_list_section("Lifestyle Suggestions",    ai.get("lifestyle_suggestions", []))
    ai_para("Follow-Up Plan",     ai.get("follow_up", ""))
    ai_para("Risk Rationale",     ai.get("risk_rationale", ""))

    # ── MODEL CONFIDENCE BAR ───────────────────────────────────────────────────
    section("Model Confidence")
    conf_pct = conf_val * 100
    filled   = int(conf_pct / 5)
    bar      = "█" * filled + "░" * (20 - filled)
    story.append(Paragraph(f"{bar}  {conf_pct:.1f}%", mono_s))
    if ai.get("confidence_note"):
        story.append(Paragraph(ai["confidence_note"], body_s))
    story.append(Spacer(1, 10))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#fecaca"), spaceAfter=6
    ))
    story.append(Paragraph(
        "<b>&#9888; MEDICAL DISCLAIMER:</b> This report is generated by an automated "
        "screening system and does not constitute a medical diagnosis. All findings must "
        "be reviewed by a qualified neurologist. No clinical decisions should be made "
        "solely on the basis of this report.",
        red_s,
    ))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY, spaceAfter=4))
    footer_tbl = Table(
        [[
            Paragraph("NeuroScan AI · Neurological Risk Assessment Platform", foot_s),
            Paragraph(f"Report ID: {d.get('report_id','')}", ftr_s),
        ]],
        colWidths=[CW * 0.6, CW * 0.4],
    )
    footer_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_tbl)

    # Build the PDF
    doc.build(story)


def _build_html(output_path: str, d: dict):
    """Fallback HTML report when ReportLab is unavailable."""
    ai  = d.get("ai_analysis", {})
    rc  = {"LOW":"#16a34a","MODERATE":"#d97706","HIGH":"#dc2626"}.get(
        d.get("risk_level",""), "#4f46e5"
    )

    def li(items):
        return "".join(f"<li>{i}</li>" for i in (items or [])) or "<li>—</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neurological Risk Assessment Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;background:#f8fafc;color:#0f172a;font-size:11pt;line-height:1.6;padding:0}}
  .page{{max-width:800px;margin:0 auto;padding:40px 40px 60px;background:#fff;box-shadow:0 0 40px rgba(0,0,0,0.08)}}

  /* HEADER — separate from body content, never overlaps */
  .report-header{{border-bottom:3px solid #0f172a;padding-bottom:18px;margin-bottom:24px}}
  .header-inner{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}}
  .org-name{{font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-0.5px}}
  .org-name span{{color:#4f46e5}}
  .report-title{{font-size:14px;font-weight:700;color:#0f172a;margin-top:4px}}
  .report-subtitle{{font-size:10px;color:#64748b;margin-top:2px}}
  .report-meta{{text-align:right;font-family:'JetBrains Mono',monospace;font-size:9px;color:#64748b;line-height:1.7;flex-shrink:0}}

  /* Patient bar */
  .patient-bar{{background:#f1f5f9;border-radius:8px;padding:10px 16px;display:flex;gap:32px;margin-bottom:20px}}
  .patient-bar span{{font-size:10px;color:#64748b}}.patient-bar b{{color:#0f172a}}

  /* NSS panel */
  .nss-panel{{background:#0f172a;color:#fff;border-radius:12px;padding:20px 24px;display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}
  .nss-block{{text-align:center}}
  .nss-label{{font-size:9px;opacity:0.55;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px}}
  .nss-val{{font-family:'JetBrains Mono',monospace;font-size:34px;font-weight:700;color:{rc}}}
  .risk-badge{{background:{rc}22;color:{rc};border:1px solid {rc}55;padding:6px 18px;border-radius:20px;font-size:13px;font-weight:700;letter-spacing:1px}}
  .z-val{{font-family:'JetBrains Mono',monospace;font-size:14px}}
  .conf-val{{font-family:'JetBrains Mono',monospace;font-size:14px;color:#a78bfa}}

  /* Formula */
  .formula{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:9.5px;color:#4f46e5;line-height:2;white-space:pre-wrap;margin:14px 0}}

  /* Section headings */
  h3{{font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#4f46e5;border-bottom:1px solid #e2e8f0;padding-bottom:5px;margin:18px 0 10px}}

  /* Metrics table */
  table{{width:100%;border-collapse:collapse;margin:8px 0 14px}}
  td{{padding:6px 10px;font-size:10px;border-bottom:1px solid #f1f5f9}}
  tr:nth-child(even){{background:#f8fafc}}
  .metric-key{{color:#64748b;font-weight:500}}
  .metric-val{{font-family:'JetBrains Mono',monospace;font-weight:600;color:#0f172a;text-align:right}}

  /* Lists */
  ul{{margin:6px 0 12px 20px;font-size:10px;line-height:1.8}}
  p{{font-size:10px;line-height:1.7;margin-bottom:10px;color:#334155}}

  /* Tags */
  .tag{{display:inline-block;background:#ede9fe;color:#5b21b6;padding:3px 10px;border-radius:12px;font-size:9px;margin:2px 3px;font-weight:600}}

  /* Confidence bar */
  .conf-bar-wrap{{background:#e2e8f0;border-radius:4px;height:8px;margin:6px 0 10px}}
  .conf-bar{{height:8px;border-radius:4px;background:linear-gradient(90deg,#4f46e5,#7c3aed)}}

  /* Disclaimer */
  .disclaimer{{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;padding:12px 16px;border-radius:8px;font-size:9px;margin-top:24px;line-height:1.6}}

  /* Footer */
  .footer{{margin-top:24px;padding-top:12px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:9px;color:#94a3b8}}
</style>
</head>
<body>
<div class="page">

  <!-- REPORT HEADER -->
  <div class="report-header">
    <div class="header-inner">
      <div>
        <div class="org-name">Neuro<span>Scan</span> AI</div>
        <div class="report-title">Neurological Risk Assessment Report</div>
        <div class="report-subtitle">Multimodal Handwriting &amp; Speech Analysis · NSS Classification</div>
      </div>
      <div class="report-meta">
        Report ID: {d.get('report_id','—')}<br>
        Date: {d.get('timestamp','')[:10]}<br>
        Time: {d.get('timestamp','')[11:19] if 'T' in d.get('timestamp','') else ''}<br>
        Version: v{settings.APP_VERSION}
      </div>
    </div>
  </div>

  <!-- PATIENT INFO -->
  <div class="patient-bar">
    <span><b>Patient:</b> {d.get('patient_name') or 'Anonymous'}</span>
    <span><b>Assessment Date:</b> {d.get('timestamp','')[:10]}</span>
    <span><b>Report ID:</b> {d.get('report_id','—')}</span>
  </div>

  <!-- NSS PANEL -->
  <div class="nss-panel">
    <div class="nss-block">
      <div class="nss-label">NSS Score</div>
      <div class="nss-val">{d.get('nss_score',0):.4f}</div>
    </div>
    <div class="nss-block">
      <div class="nss-label">Risk Level</div>
      <div class="risk-badge">{d.get('risk_emoji','')} {d.get('risk_level','—')}</div>
    </div>
    <div class="nss-block">
      <div class="nss-label">Z-Score</div>
      <div class="z-val">{d.get('z_score',0):.5f}</div>
    </div>
    <div class="nss-block">
      <div class="nss-label">Confidence</div>
      <div class="conf-val">{d.get('confidence',0)*100:.1f}%</div>
    </div>
  </div>

  <h3>Computation Pipeline</h3>
  <div class="formula">{d.get('formula_display','')}</div>

  <h3>Handwriting Stroke Metrics</h3>
  <table>
    {"".join(f'<tr><td class="metric-key">{k.replace("_"," ").title()}</td><td class="metric-val">{v:.4f if isinstance(v,float) else v}</td></tr>' for k,v in d.get("stroke_metrics",{}).items())}
  </table>

  <h3>Acoustic Speech Metrics</h3>
  <table>
    {"".join(f'<tr><td class="metric-key">{k.replace("_"," ").title()}</td><td class="metric-val">{v:.4f if isinstance(v,float) else v}</td></tr>' for k,v in d.get("acoustic_metrics",{}).items())}
  </table>

  <h3>Clinical Summary</h3>
  <p>{ai.get('clinical_summary','')}</p>

  <h3>Handwriting Findings</h3>
  <p>{ai.get('handwriting_findings','')}</p>

  <h3>Speech Findings</h3>
  <p>{ai.get('speech_findings','')}</p>

  <h3>Neurological Indicators</h3>
  <ul>{li(ai.get('neurological_indicators',[]))}</ul>

  <h3>Differential Diagnosis</h3>
  <p>{"".join(f'<span class="tag">{item}</span>' for item in ai.get('differential_diagnosis',[]))}</p>

  <h3>Clinical Recommendations</h3>
  <ul>{li(ai.get('recommendations',[]))}</ul>

  <h3>Lifestyle Suggestions</h3>
  <ul>{li(ai.get('lifestyle_suggestions',[]))}</ul>

  <h3>Follow-Up Plan</h3>
  <p>{ai.get('follow_up','')}</p>

  <h3>Risk Rationale</h3>
  <p>{ai.get('risk_rationale','')}</p>

  <h3>Model Confidence</h3>
  <div class="conf-bar-wrap"><div class="conf-bar" style="width:{d.get('confidence',0)*100:.0f}%"></div></div>
  <p>{ai.get('confidence_note','')}</p>

  <div class="disclaimer">
    <b>&#9888; Medical Disclaimer:</b> This report is generated by an automated screening system
    for preliminary assessment purposes only. It does not constitute a medical diagnosis.
    All findings must be reviewed and interpreted by a qualified neurologist before any
    clinical decision is made.
  </div>

  <div class="footer">
    <span>NeuroScan AI · Neurological Risk Assessment Platform</span>
    <span>Report ID: {d.get('report_id','')}</span>
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


class ReportService:
    """Generates downloadable PDF (with ReportLab fallback to HTML)."""

    def generate(self, report_data: dict, output_path: str) -> str:
        # Ensure the output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            _build_pdf(output_path, report_data)
            logger.info(f"ReportService: PDF generated → {output_path}")
            return output_path
        except ImportError:
            logger.warning("ReportLab not installed — falling back to HTML report.")
            html_path = output_path.replace(".pdf", ".html")
            _build_html(html_path, report_data)
            logger.info(f"ReportService: HTML report generated → {html_path}")
            return html_path
        except Exception as exc:
            logger.error(f"ReportService: PDF failed ({exc}) — falling back to HTML.")
            html_path = output_path.replace(".pdf", ".html")
            _build_html(html_path, report_data)
            return html_path
