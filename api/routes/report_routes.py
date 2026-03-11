"""
ForenSight AI — Report Download Routes
Generates and serves PDF analysis reports.
"""

import traceback
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.auth import get_current_user
from api.database import get_analysis_by_id

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("/{analysis_id}")
async def download_report(analysis_id: str, user=Depends(get_current_user)):
    """Generate and download a PDF report for an analysis."""

    # Fetch analysis from database (also verifies ownership)
    try:
        analysis = await get_analysis_by_id(analysis_id, user["user_id"])
    except Exception as e:
        print(f"[Report] DB error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    # Generate PDF
    try:
        from api.report_generator import generate_analysis_pdf

        heatmap = analysis.get("heatmap_path")  # base64 or None
        pdf_bytes = generate_analysis_pdf(analysis, heatmap_b64=heatmap)

        file_name = analysis.get("file_name", "analysis").rsplit(".", 1)[0]
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in file_name)

        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="ForenSight_Report_{safe_name}.pdf"'
            },
        )
    except Exception as e:
        print(f"[Report] PDF generation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
