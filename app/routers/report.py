"""
app/routers/report.py
POST /api/v1/report/{report_id} — generates and streams PDF report.
"""

import os, logging, tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.routers.analysis import result_cache
from app.services.report_service import ReportService

logger = logging.getLogger(__name__)
router = APIRouter()
report_svc = ReportService()


@router.get("/report/{report_id}")
async def download_report(report_id: str):
    """Generate and download the PDF/HTML report for a completed assessment."""
    if report_id not in result_cache:
        raise HTTPException(404, f"Report {report_id!r} not found.")

    data = result_cache[report_id]

    # Build flat report_data dict that ReportService expects
    risk = data.get("risk", {})
    nss  = data.get("nss_computation", {})
    report_data = {
        "report_id":      data["report_id"],
        "patient_name":   data.get("patient_name", "Anonymous"),
        "timestamp":      data["timestamp"],
        "nss_score":      nss.get("nss_score", 0.0),
        "z_score":        nss.get("z_score", 0.0),
        "formula_display": nss.get("formula_display", data.get("formula_display", "")),
        "risk_level":     risk.get("level", "UNKNOWN"),
        "risk_emoji":     risk.get("emoji", ""),
        "confidence":     risk.get("confidence_score", 0.0),
        "stroke_metrics": data.get("stroke_metrics", {}),
        "acoustic_metrics": data.get("acoustic_metrics", {}),
        "ai_analysis":    data.get("ai_analysis", {}),
    }

    # Write to temp file
    suffix = ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()

    try:
        out_path = report_svc.generate(report_data, tmp.name)
        media_type = "application/pdf" if out_path.endswith(".pdf") else "text/html"
        ext        = ".pdf" if media_type == "application/pdf" else ".html"
        filename   = f"NeuroScan_{(data.get('patient_name') or 'Patient').replace(' ','_')}_{report_id}{ext}"
        logger.info(f"Serving report: {out_path}")
        return FileResponse(
            path=out_path,
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(500, f"Report generation failed: {e}")
