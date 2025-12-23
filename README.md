# Automatic Parking Management System

An end-to-end parking assistant that combines two capabilities:

- **License plate recognition** - detect vehicles and read their plates using YOLO + EasyOCR.
- **Parking recommendation** - highlight the closest available parking spot and visualize the driving path.

The entire business logic lives in the `backend/` FastAPI service. CLI scripts in `scripts/` provide lightweight wrappers for quick local testing.

## Repository layout

```
smart-paking-ai/
├── backend/              # FastAPI application + shared services/config
│   ├── app/
│   │   ├── config.py     # Centralized configuration
│   │   ├── routers/      # HTTP endpoints
│   │   ├── services/     # YOLO + OCR pipelines
│   │   ├── models/       # Pydantic schemas
│   │   └── utils/        # Utilities (MongoDB, images)
│   ├── main.py           # FastAPI entry point
│   └── requirements.txt  # Python dependencies
├── models/               # Centralized YOLO model weights
│   ├── parking/
│   │   └── best.pt
│   └── license_plate/
│       └── best.pt
├── scripts/              # CLI tools for local testing
│   ├── license_plate.py  # License plate detection CLI
│   └── parking_space.py  # Parking recommendation CLI
└── tests/                # Test data and outputs
    ├── license_plate/
    └── parking_space/
```

## Quick Start

### Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   # Windows
   run_server.bat
   # Linux/Mac
   chmod +x run_server.sh
   ./run_server.sh
   # Or manually:
   uvicorn backend.main:app --reload
   ```

The server will start at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## CLI usage

Both CLI scripts delegate to the backend services but run without HTTP:

```bash
# License plate detection
python scripts/license_plate.py path/to/video.mp4

# Parking recommendation
python scripts/parking_space.py path/to/video.mp4
```

Use `--output-dir <path>` to store debug frames somewhere else (defaults to `tests/license_plate/` or `tests/parking_space/`).

## HTTP endpoints

- `POST /parking-space/recommend` – upload an image or video to receive a single annotated PNG showing the closest empty slot.
- `POST /parking-space/recommend-video` – same as above but dedicated to video form-data.
- `POST /parking-space/annotate-video` – upload a full video to receive an MP4 with every frame annotated (ideal for “real-time” playback after processing).
- `POST /license-plate/detect` – upload an image to get an annotated PNG and recognized plate text (via response header).
- `GET /license-plate/logs` – retrieve all logged license plates.

### Performance tuning

- The parking pipeline automatically downscales frames to `parking_max_width` (default 960 px) before running YOLO. Adjust `backend/app/config.py` if you want higher accuracy or lower latency.
- Endpoints that accept video expose `frame_stride` – increasing this value skips frames (re-using the last annotation for skipped ones) and drastically reduces processing time.
- Real-time endpoints now include a lightweight centroid tracker so bounding boxes persist smoothly between detections; when skipping frames, the tracker keeps positions stable.
