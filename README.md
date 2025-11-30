# Automatic Parking Management System

An end-to-end parking assistant that combines two capabilities:

- **License plate recognition** - detect vehicles and read their plates using YOLO + EasyOCR.
- **Parking recommendation** - highlight the closest available parking spot and visualize the driving path.

The entire business logic now lives in the `backend/` FastAPI service. The legacy `LicensePlate/` and `ParkingSpace/` folders only provide lightweight CLI wrappers that reuse the backend services for quick local testing.

## Repository layout

```
backend/               # FastAPI application + shared services/config
  app/
    config.py
    routers/           # HTTP endpoints
    services/          # YOLO + OCR pipelines
    utils/
LicensePlate/          # CLI wrapper around backend license-plate service
ParkingSpace/          # CLI wrapper around backend parking service
models/                # YOLO weights for each task
```

## Setup

1. Create a virtual environment (optional) and install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```

## CLI usage

Both CLI scripts now delegate to the backend services but run without HTTP:

```bash
# License plate detection
python LicensePlate/app.py path/to/video.mp4

# Parking recommendation
python ParkingSpace/app.py path/to/video.mp4
```

Use `--output-dir <path>` to store debug frames somewhere else (defaults to each module's `tests/` directory).

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
