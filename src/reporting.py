"""
ProjectIQ V2.0 — Automated PDF Decision Brief Generator
Produces a downloadable, structured decision document using ReportLab.
"""

import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

def clean_markdown_to_reportlab(text: str) -> list:
    """
    Converts markdown text safely into a list of styled ReportLab paragraphs.
    """
    lines = text.split("\n")
    flowables = []
    styles = getSampleStyleSheet()
    
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["Normal"],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#334155")
    )
    h3_style = ParagraphStyle(
        "ReportH3", parent=styles["Heading3"],
        fontSize=10.5, leading=14, textColor=colors.HexColor("#0F172A"),
        spaceBefore=6, spaceAfter=2
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for headings
        if line.startswith("### "):
            header_text = line.replace("### ", "").strip()
            flowables.append(Paragraph(f"<b>{header_text}</b>", h3_style))
        else:
            # Convert markdown bold **text** to <b>text</b>
            formatted = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            # Convert markdown italic *text* to <i>text</i>
            formatted = re.sub(r"\*(.*?)\*", r"<i>\1</i>", formatted)
            flowables.append(Paragraph(formatted, body_style))
            flowables.append(Spacer(1, 2))

    return flowables

class DecisionReportGenerator:
    @staticmethod
    def generate_pdf(payload: dict, output_filename: str = "ProjectIQ_Decision_Brief.pdf") -> str:
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        story = []
        styles = getSampleStyleSheet()

        # Typography styles
        title_style = ParagraphStyle(
            "TitleStyle", parent=styles["Heading1"],
            fontSize=18, leading=22, textColor=colors.HexColor("#1E293B")
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle", parent=styles["Normal"],
            fontSize=8.5, leading=11, textColor=colors.HexColor("#64748B")
        )
        h2_style = ParagraphStyle(
            "H2Style", parent=styles["Heading2"],
            fontSize=11.5, leading=15, textColor=colors.HexColor("#0F172A"),
            spaceBefore=6, spaceAfter=4
        )
        body_style = ParagraphStyle(
            "BodyStyle", parent=styles["Normal"],
            fontSize=8.5, leading=12, textColor=colors.HexColor("#334155")
        )
        bold_body = ParagraphStyle("BoldBody", parent=body_style, fontName="Helvetica-Bold")

        # Header Section
        story.append(Paragraph("<b>ProjectIQ: Project Risk Assessment & Decision Brief</b>", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version: V2.0 Decision Intelligence | Prediction Point: T₀ (Pre-Launch)",
            subtitle_style
        ))
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=8))

        # Project Summary Table
        summary_data = [
            [
                Paragraph("<b>Project Name:</b>", body_style), Paragraph(payload.get("name", "N/A"), body_style),
                Paragraph("<b>Category:</b>", body_style), Paragraph(payload.get("category", "N/A"), body_style)
            ],
            [
                Paragraph("<b>Target Goal:</b>", body_style), Paragraph(f"${payload.get('goal_usd', 0):,.0f} USD", body_style),
                Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{payload.get('duration_days', 0)} Days", body_style)
            ],
            [
                Paragraph("<b>Predicted Probability:</b>", bold_body), Paragraph(f"<b>{payload.get('success_probability', 0):.1f}%</b>", bold_body),
                Paragraph("<b>Risk Tier:</b>", bold_body), Paragraph(f"<b>{payload.get('risk_tier', 'N/A')}</b>", bold_body)
            ]
        ]
        t_summary = Table(summary_data, colWidths=[110, 160, 100, 170])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 6))

        # SHAP Waterfall Section
        story.append(Paragraph("1. Explainable Risk Drivers (SHAP Local Attribution)", h2_style))
        if os.path.exists("test_waterfall.png"):
            story.append(Image("test_waterfall.png", width=480, height=200))
        story.append(Spacer(1, 4))

        # Risk Register Table
        story.append(Paragraph("2. Project Risk Register & Action Plan", h2_style))
        reg_headers = [
            Paragraph("<b>ID</b>", bold_body),
            Paragraph("<b>Risk Factor</b>", bold_body),
            Paragraph("<b>Evidence Detail</b>", bold_body),
            Paragraph("<b>Recommended Response</b>", bold_body)
        ]
        reg_rows = [reg_headers]
        for rsk in payload.get("risk_register", [])[:4]:
            reg_rows.append([
                Paragraph(rsk.get("risk_id", ""), body_style),
                Paragraph(rsk.get("risk_title", ""), bold_body),
                Paragraph(rsk.get("evidence_detail", ""), body_style),
                Paragraph(rsk.get("suggested_response", ""), body_style)
            ])
        t_reg = Table(reg_rows, colWidths=[45, 120, 185, 190])
        t_reg.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_reg)
        story.append(Spacer(1, 6))

        # Narrative Section
        story.append(Paragraph("3. Executive Advisory & Next Steps", h2_style))
        narrative_paragraphs = clean_markdown_to_reportlab(payload.get("narrative", ""))
        for p in narrative_paragraphs:
            story.append(p)
        story.append(Spacer(1, 6))

        # Footer & Academic Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=4))
        story.append(Paragraph(
            "<b>Academic Disclaimer:</b> ProjectIQ is an AI decision-support prototype. Estimates represent statistical associations from historical data and do not constitute guaranteed outcomes.",
            subtitle_style
        ))

        doc.build(story)
        return output_filename