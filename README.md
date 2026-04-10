# Sign Language Translator

This project includes:
- Python backend (FastAPI + model training/inference),
- React frontend (Vite + TypeScript + SCSS Modules),
- keypoints dataset slovo_keypoints is expected.

## What is in your files

- `annotations.csv` - metadata and ground-truth label (`text`) for each `attachment_id`.
- `slovo_keypoints/*.npy` - full keypoints set .

Each `.npy` file is a NumPy array saved to disk. In this project it stores a tensor of shape `155 x 2 x 21 x 3`:
- `155` frames,
- `2` hands,
- `21` hand landmarks,
- `3` coordinates (`x`, `y`, `z`).

## Why convert video to keypoints

Raw video is large and contains a lot of irrelevant information (background, lighting, clothes).
For sign recognition, model usually needs hand/body landmarks only.

So the common pipeline is:
1. video -> keypoints (MediaPipe),
2. keypoints -> classifier/neural model.

This is exactly what this app does during inference.

## How to create more `.npy` files from a full dataset

If you have the raw videos from Kaggle, you can generate more keypoints with:

```powershell
python prepare_keypoints.py --videos-dir PATH_TO_RAW_VIDEOS --output-dir keypoints_all --use-begin-end
```

Use `--use-begin-end` when your videos are longer clips and `annotations.csv` defines the gesture range.
If you only have the 100 `.npy` files and no raw videos, then more samples cannot be reconstructed from those files alone.

## Backend Quick Start

1. Create and activate virtual environment (if needed):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the backend:

```powershell
python main.py
```

4. Backend URL:

- http://localhost:8000

## Frontend (Vite + React TS)

1. Install frontend dependencies:

```powershell
cd frontend
pnpm install
```

2. Development mode:

```powershell
pnpm dev
```

3. Production build (served by FastAPI from frontend/dist):

```powershell
pnpm build
cd ..
python main.py
```

## Usage

1. Click Train with slovo_keypoints once.
2. Use either:
- Upload Video, or
- Record Camera.
3. Click predict and check output text + confidence.

## Notes

- Training can take noticeable time.
- If frontend build is missing, backend root endpoint will ask you to run pnpm build.
- `prepare_keypoints.py` is needed only if you want to regenerate keypoints from raw videos.

## Files

- `main.py` - FastAPI server and endpoints.
- `signer_model.py` - training + video keypoint extraction + prediction logic.
- `frontend/` - Vite React TS app with SCSS Modules.
