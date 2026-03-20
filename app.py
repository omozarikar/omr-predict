import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from omr_pipeline import process_omr_image_file, ensure_folder

# ==========================================
# SETTINGS
# ==========================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROW_MODEL_PATH = BASE_DIR / "sheet_model" / "best.pt"
BUBBLE_MODEL_PATH = BASE_DIR / "bubble_model" / "best.pt"

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
DEBUG_DIR = "debug_outputs"

ensure_folder(UPLOAD_DIR)
ensure_folder(OUTPUT_DIR)
ensure_folder(DEBUG_DIR)


# ==========================================
# LOAD MODELS ON STARTUP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.row_model = YOLO(ROW_MODEL_PATH)
    app.state.bubble_model = YOLO(BUBBLE_MODEL_PATH)
    yield


app = FastAPI(
    title="OMR Detection API",
    version="1.0.0",
    lifespan=lifespan
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
    ext = os.path.splitext(file.filename)[1].lower()
    allowed = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    temp_input_path = os.path.join(UPLOAD_DIR, file.filename)
    csv_output_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(file.filename)[0] + "_answers.csv"
    )

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_omr_image_file(
            image_path=temp_input_path,
            row_model=app.state.row_model,
            bubble_model=app.state.bubble_model,
            output_csv=csv_output_path if export_csv else None,
            debug_folder=DEBUG_DIR,
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
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


@app.get("/download-csv/{csv_name}")
def download_csv(csv_name: str):
    csv_path = os.path.join(OUTPUT_DIR, csv_name)

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not found")

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=csv_name
    )