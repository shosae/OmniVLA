"""FastAPI wrapper for the shared fp16, AWQ, and TensorRT OmniVLA inference path."""

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

import inference as omnivla


inference: Optional[omnivla.Inference] = None
backend = ""
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference, backend
    inference, config = omnivla.create_inference()
    backend = config.backend
    yield
    inference = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if inference is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return {"status": "ready", "backend": backend}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    instruction: Optional[str] = Form(None),
    waypoint_select: int = Form(4),
):
    if inference is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if not 0 <= waypoint_select < 8:
        raise HTTPException(status_code=422, detail="waypoint_select must be between 0 and 7")

    try:
        current_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="image must be a readable RGB image") from exc

    async with inference_lock:
        torch.cuda.synchronize(inference.device_id)
        started = time.perf_counter()
        linear, angular = inference.run_omnivla(
            waypoint_select=waypoint_select,
            current_image=current_image,
            instruction=instruction,
        )
        torch.cuda.synchronize(inference.device_id)

    return {
        "backend": backend,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "llm_inference_ms": round(inference.llm_backend.last_llm_inference_ms, 3),
        "llm_memory_mib": inference.llm_backend.last_llm_memory_mib,
        "linear_velocity": float(linear),
        "angular_velocity": float(angular),
        "waypoints": inference.last_waypoints,
        "goal_pose": inference.last_goal_pose,
        "modality_id": inference.last_modality_id,
        "waypoint_select": inference.last_waypoint_select,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("OMNIVLA_HOST", "0.0.0.0"), port=int(os.environ.get("OMNIVLA_PORT", "8000")))
