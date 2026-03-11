import sys
import os
sys.path.append(os.path.abspath('e:/Deepfake/AntiGravity-Deepfake-Detection'))

from api.report_generator import generate_analysis_pdf

analysis = {
    "analysis_id": "test-123",
    "user_id": "user-123",
    "file_name": "test_image.jpg",
    "file_type": "image",
    "result": "FAKE",
    "confidence_score": 0.85,
    "deepfake_probability": 0.85,
    "cnn_score": 0.85,
    "vit_score": 0.85,
    "temporal_score": None,
    "model_agreement": 0.85,
    "face_detected": True,
    "frames_analyzed": None,
    "processing_time_sec": 1.5,
    "heatmap_path": None,
    "created_at": "2026-03-10T00:00:00"
}

try:
    pdf_bytes = generate_analysis_pdf(analysis)
    print("SUCCESS")
    print(f"Generated {len(pdf_bytes)} bytes")
except Exception as e:
    import traceback
    traceback.print_exc()
