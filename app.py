import os
import shutil
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from omr_pipeline import process_omr_image_file, ensure_folder


import torch
import ultralytics

print("TORCH VERSION:", torch.__version__)
print("ULTRALYTICS VERSION:", ultralytics.__version__)
# ==========================================
# SETTINGS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DEBUG_DIR = BASE_DIR / "debug_outputs"

ensure_folder(str(UPLOAD_DIR))
ensure_folder(str(OUTPUT_DIR))
ensure_folder(str(DEBUG_DIR))

process_lock = asyncio.Lock()

app = FastAPI(
    title="OMR Detection API",
    version="1.0.0"
)

# ===============================
# CORS CONFIGURATION
# ===============================
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "OMR backend running"}


@app.post("/process-omr")
async def process_omr(
    file: UploadFile = File(...),
    save_debug: bool = Form(False),
    export_csv: bool = Form(True)
):
    if process_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Server is busy processing another OMR file. Please try again."
        )

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    temp_input_path = UPLOAD_DIR / file.filename
    csv_output_path = OUTPUT_DIR / (Path(file.filename).stem + "_answers.csv")

    async with process_lock:
        try:
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            result = process_omr_image_file(
                image_path=str(temp_input_path),
                output_csv=str(csv_output_path) if export_csv else None,
                debug_folder=str(DEBUG_DIR),
                save_debug=save_debug
            )

            return {
                "status": "success",
                "filename": file.filename,
                "total_rows_detected": result["total_rows_detected"],
                "results": result["results"],
                "csv_path": result["csv_path"]
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if temp_input_path.exists():
                temp_input_path.unlink()


@app.get("/download-csv/{csv_name}")
def download_csv(csv_name: str):
    csv_path = OUTPUT_DIR / csv_name

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")

    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=csv_name
    )