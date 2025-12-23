@echo off
echo Starting Smart Parking Backend Server...
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

