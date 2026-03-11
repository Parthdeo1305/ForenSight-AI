"""
ForenSight AI — PDF Report Generator
Creates professional, branded PDF reports for deepfake analysis results.

Uses fpdf2 (pure Python, lightweight).

Author: ForenSight AI Team
"""

import io
import os
import base64
from datetime import datetime
from typing import Dict, Optional

from fpdf import FPDF


# ─── Brand Colors (RGB) ───────────────────────────────────────────────────────

PURPLE = (124, 58, 237)
DARK_BG = (15, 15, 25)
CARD_BG = (25, 25, 40)
RED = (239, 68, 68)
GREEN = (16, 185, 129)
CYAN = (6, 182, 212)
GRAY = (156, 163, 175)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 210)


class ForenSightReport(FPDF):
    """Custom PDF with ForenSight AI branding."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        # Purple header bar
        self.set_fill_color(*PURPLE)
        self.rect(0, 0, 210, 28, 'F')

        # Title
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*WHITE)
        self.set_xy(10, 6)
        self.cell(0, 8, "ForenSight AI", ln=False)

        # Subtitle
        self.set_font("Helvetica", "", 8)
        self.set_xy(10, 16)
        self.cell(0, 6, "Digital Deepfake Detection Platform  |  Analysis Report")

        # Date on right
        self.set_font("Helvetica", "", 8)
        self.set_xy(150, 6)
        self.cell(0, 8, datetime.utcnow().strftime("%B %d, %Y"))
        self.ln(30)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"ForenSight AI  |  Confidential Report  |  Page {self.page_no()}", align="C")

    def _section_title(self, title: str, color=PURPLE):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*color)
        self.cell(0, 10, title, ln=True)
        # Accent line
        self.set_draw_color(*color)
        self.set_line_width(0.8)
        self.line(self.get_x(), self.get_y(), self.get_x() + 50, self.get_y())
        self.ln(4)

    def _key_value(self, key: str, value: str, bold_value=False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(65, 6, key, ln=False)
        self.set_font("Helvetica", "B" if bold_value else "", 9)
        self.set_text_color(*WHITE if bold_value else LIGHT_GRAY)
        self.cell(0, 6, str(value), ln=True)

    def _draw_score_bar(self, label: str, score: float, color: tuple):
        """Draw a horizontal score bar with label."""
        y = self.get_y()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(65, 7, label, ln=False)

        # Bar background
        bar_x = self.get_x()
        bar_width = 90
        bar_height = 5
        self.set_fill_color(40, 40, 55)
        self.rect(bar_x, y + 1, bar_width, bar_height, 'F')

        # Bar fill
        fill_width = bar_width * min(score, 1.0)
        self.set_fill_color(*color)
        self.rect(bar_x, y + 1, fill_width, bar_height, 'F')

        # Score text
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.set_xy(bar_x + bar_width + 3, y)
        self.cell(20, 7, f"{round(score * 100)}%", ln=True)
        self.ln(2)

    def _draw_verdict_box(self, is_fake: bool, confidence: float):
        """Large verdict box with color coding."""
        y = self.get_y()
        color = RED if is_fake else GREEN
        label = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC MEDIA"
        emoji = "WARNING" if is_fake else "VERIFIED"

        # Box
        self.set_fill_color(*color)
        self.rect(10, y, 190, 30, 'F')

        # Verdict text
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*WHITE)
        self.set_xy(15, y + 4)
        self.cell(0, 12, f"{emoji}: {label}", ln=True)

        # Confidence
        self.set_font("Helvetica", "", 10)
        self.set_xy(15, y + 18)
        self.cell(0, 8, f"System Confidence: {round(confidence * 100)}%")
        self.set_y(y + 35)


def generate_analysis_pdf(analysis: Dict, heatmap_b64: Optional[str] = None) -> bytes:
    """
    Generate a professional PDF report for a deepfake analysis result.
    
    Args:
        analysis: Analysis result dictionary from the database
        heatmap_b64: Optional base64-encoded heatmap image
        
    Returns:
        PDF file as bytes
    """
    pdf = ForenSightReport()
    pdf.set_draw_color(*GRAY)
    pdf.add_page()

    # ── Verdict Box ─────────────────────────────────────────────────────
    is_fake = (analysis.get("result") or "").upper() == "FAKE"
    confidence = analysis.get("confidence_score") or analysis.get("deepfake_probability") or 0

    pdf._draw_verdict_box(is_fake, confidence)
    pdf.ln(4)

    # ── File Information ────────────────────────────────────────────────
    pdf._section_title("File Information")
    pdf._key_value("File Name:", str(analysis.get("file_name") or "Unknown"), bold_value=True)
    pdf._key_value("File Type:", str(analysis.get("file_type") or "Unknown").upper(), bold_value=True)
    pdf._key_value("Analysis ID:", str(analysis.get("analysis_id") or "N/A"))
    pdf._key_value("Analysis Date:", str(analysis.get("created_at") or "N/A"))
    pt = analysis.get("processing_time_sec")
    pdf._key_value("Processing Time:", f"{pt} seconds" if pt else "N/A")
    face = analysis.get("face_detected")
    pdf._key_value("Face Detected:", "Yes" if face else "No" if face is not None else "N/A")
    frames = analysis.get("frames_analyzed")
    if frames:
        pdf._key_value("Frames Analyzed:", str(frames))
    pdf.ln(6)

    # ── Detection Results ───────────────────────────────────────────────
    pdf._section_title("Detection Results")

    prob = analysis.get("deepfake_probability") or 0
    pdf._key_value("Verdict:", "FAKE (Deepfake)" if is_fake else "REAL (Authentic)", bold_value=True)
    pdf._key_value("Deepfake Probability:", f"{round(prob * 100, 1)}%", bold_value=True)
    pdf._key_value("Confidence Score:", f"{round(confidence * 100, 1)}%", bold_value=True)
    agreement = analysis.get("model_agreement")
    if agreement:
        pdf._key_value("Model Agreement:", f"{round(agreement * 100, 1)}%")
    pdf.ln(6)

    # ── Per-Model Scores ────────────────────────────────────────────────
    pdf._section_title("Per-Model Analysis Scores")

    cnn = analysis.get("cnn_score")
    vit = analysis.get("vit_score")
    temporal = analysis.get("temporal_score")

    if cnn is not None:
        pdf._draw_score_bar("EfficientNet-B0 (Spatial CNN)", cnn, PURPLE)
    if vit is not None:
        pdf._draw_score_bar("ViT-B/16 (Vision Transformer)", vit, CYAN)
    if temporal is not None:
        pdf._draw_score_bar("CNN+LSTM (Temporal)", temporal, GREEN)
    pdf.ln(4)

    # ── Heatmap ─────────────────────────────────────────────────────────
    if heatmap_b64:
        try:
            # Decode base64 heatmap
            if "," in heatmap_b64:
                heatmap_b64 = heatmap_b64.split(",")[1]
            img_data = base64.b64decode(heatmap_b64)
            
            # Save temp image
            import tempfile
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(img_data)

            pdf._section_title("Grad-CAM Explainability Heatmap")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*GRAY)
            pdf.cell(0, 5, "Visual evidence: Red/Yellow regions highlight the specific manipulated or AI-generated areas.", ln=True)
            pdf.cell(0, 5, "Blue regions indicate areas that had low influence on the AI's final decision.", ln=True)
            pdf.ln(4)
            
            # Center the image
            pdf.image(tmp_path, x=30, w=150)
            pdf.ln(8)

            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            print(f"[ReportGen] Heatmap embed failed: {e}")

    # ── Detailed Analysis ───────────────────────────────────────────────
    pdf._section_title("Detailed Analysis")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*LIGHT_GRAY)
    
    # Increase line height for readability
    line_height = 6

    if is_fake:
        if confidence > 0.80:
            verdict_text = "Analysis strongly indicates that this media has been digitally altered or AI-generated."
        else:
            verdict_text = "Analysis detected suspicious signs of digital manipulation, though the evidence is not overwhelmingly strong. Proceed with caution."
            
        analysis_text = (
            f"OVERALL VERDICT:\n"
            f"{verdict_text}\n\n"
            f"TECHNICAL BREAKDOWN:\n"
            f"1. Spatial Analysis (EfficientNet): Looked for warped pixels, irregular skin textures, and edge blending artifacts frame-by-frame.\n"
            f"2. Global Context (Vision Transformer): Analyzed the entire image for lighting, shadow inconsistencies, and overarching frequency anomalies.\n"
        )
        if temporal is not None:
             analysis_text += f"3. Temporal Consistency (LSTM): Watched the video over time to catch unnatural movements or micro-flickers between frames.\n"
        
        analysis_text += (
            f"\nCONCLUSION & RECOMMENDATION:\n"
            f"Our AI ensemble detected mathematical patterns consistent with known deepfake signatures "
            f"(such as FaceSwap, Face2Face, or generative techniques). We highly recommend treating this media as fabricated or heavily edited. "
            f"Do not use this media as a basis for factual or journalistic claims."
        )
    else:
        if confidence > 0.80:
            verdict_text = "This media appears to be completely authentic. Our systems found no traces of digital manipulation."
        else:
            verdict_text = "This media likely hasn't been altered, but our confidence is somewhat low due to poor image quality, heavy compression, or unusual lighting."
            
        analysis_text = (
            f"OVERALL VERDICT:\n"
            f"{verdict_text}\n\n"
            f"TECHNICAL BREAKDOWN:\n"
            f"The analyzed media exhibits natural textures, mathematically consistent lighting, and expected frequency-domain characteristics. "
            f"None of our specialized AI models (Spatial CNN, Vision Transformer, or Temporal LSTM) flagged any significant anomalies "
            f"associated with face-swapping, deepfaking, or generative AI synthesis.\n\n"
            f"CONCLUSION & RECOMMENDATION:\n"
            f"Based on the extracted visual features, this media is classified as real. However, please note that no AI detection system is 100% infallible. "
            f"Extremely advanced or novel deepfakes could potentially evade detection. Always consider the source and visual context of the media."
        )

    pdf.multi_cell(0, line_height, analysis_text)
    pdf.ln(8)

    # ── Disclaimer ──────────────────────────────────────────────────────
    pdf._section_title("Disclaimer", color=GRAY)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 4, (
        "This report is generated by ForenSight AI and is provided for informational purposes only. "
        "The analysis is based on machine learning models which may not be 100% accurate. "
        "This report should not be used as the sole basis for legal, journalistic, or other "
        "critical decisions. ForenSight AI makes no warranties regarding the accuracy or "
        "completeness of this analysis. Always verify findings with multiple sources."
    ))

    # Output
    return pdf.output()
