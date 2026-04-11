from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from signer_model import SignTranslatorModel

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Sign Language Translator")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

translator = SignTranslatorModel(
	annotations_path=BASE_DIR / "annotations.csv",
	keypoints_dir=BASE_DIR / "slovo_keypoints",
	model_path=BASE_DIR / "models" / "model.joblib",
	encoder_path=BASE_DIR / "models" / "label_encoder.joblib",
)

assets_dir = FRONTEND_DIST_DIR / "assets"
if assets_dir.exists():
	app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
def root() -> FileResponse:
	index_path = FRONTEND_DIST_DIR / "index.html"
	if not index_path.exists():
		raise HTTPException(
			status_code=404,
			detail="Frontend build not found. Run pnpm install and pnpm build in frontend folder.",
		)
	return FileResponse(index_path)


@app.get("/health")
def health() -> dict:
	model_ready = translator.load_if_exists() if translator.model is None else True
	return {"status": "ok", "model_ready": model_ready}


@app.post("/api/train")
def train_model() -> dict:
	try:
		result = translator.train()
	except Exception as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	return {
		"message": "Model trained successfully",
		"samples": result.samples,
		"classes": result.classes,
		"train_accuracy": result.train_accuracy,
		"test_accuracy": result.test_accuracy,
		"total_matched_samples": result.total_matched_samples,
		"skipped_non_letters": result.skipped_non_letters,
		"label_mode": "letters-only",
	}


def _pack_prediction_response(
	label: str,
	confidence: float,
	top_predictions: list[tuple[str, float]],
) -> dict:
	return {
		"prediction": label,
		"confidence": confidence,
		"top_k": [
			{"label": candidate_label, "confidence": candidate_confidence}
			for candidate_label, candidate_confidence in top_predictions
		],
	}


@app.post("/api/predict-video")
async def predict_video(
	file: UploadFile = File(...),
	top_k: int = Query(default=5, ge=1, le=10),
) -> dict:
	suffix = Path(file.filename or "video.webm").suffix or ".webm"

	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		data = await file.read()
		tmp.write(data)
		tmp_path = Path(tmp.name)

	try:
		label, confidence, top_predictions = translator.predict_from_video(tmp_path, top_k=top_k)
	except Exception as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	finally:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)

	return _pack_prediction_response(label, confidence, top_predictions)


@app.post("/api/predict-keypoints")
async def predict_keypoints(
	file: UploadFile = File(...),
	top_k: int = Query(default=5, ge=1, le=10),
) -> dict:
	if not (file.filename or "").endswith(".npy"):
		raise HTTPException(status_code=400, detail="Upload a .npy file")

	with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
		data = await file.read()
		tmp.write(data)
		tmp_path = Path(tmp.name)

	try:
		keypoints = np.load(tmp_path)
		label, confidence, top_predictions = translator.predict_from_keypoints(
			translator._coerce_shape(keypoints),
			top_k=top_k,
		)
	except Exception as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	finally:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)

	return _pack_prediction_response(label, confidence, top_predictions)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
