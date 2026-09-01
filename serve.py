"""담다 AI 추론 HTTP 서버 — DamdaInferenceModel 을 POST /infer 로 노출.

BE(웹 백엔드)가 이 서버를 호출해 실제 분석 결과를 받는다. torch/모델이 무거우니
웹 BE와 분리해 이 서버만 GPU/RAM 충분한 곳에서 실행한다.

실행:
    set DAMDA_CKPT=checkpoints_v6macro/best.pt
    set DAMDA_CONFIG=configs/macro.yaml
    uvicorn serve:app --host 0.0.0.0 --port 9000

의존: fastapi, uvicorn, python-multipart + (AI 학습 의존성: torch 등)
"""
from __future__ import annotations

import io
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from src.infer import DamdaInferenceModel

CKPT = os.getenv("DAMDA_CKPT", "checkpoints_v6macro/best.pt")
CONFIG = os.getenv("DAMDA_CONFIG", "configs/macro.yaml")

app = FastAPI(title="담다 AI 추론 서버")
_model: DamdaInferenceModel | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _model
    _model = DamdaInferenceModel(CKPT, config_path=CONFIG)


@app.get("/health")
def health():
    return {"status": "ok", "ckpt_epoch": _model.ckpt_epoch if _model else None}


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    region: str = Form("PART_0"),
    sensor: str | None = Form(None),   # JSON 문자열 (선택)
):
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다")

    sensor_dict = json.loads(sensor) if sensor else None
    return _model.predict(
        image_path=img,
        region=region,
        sensor=sensor_dict,
        return_probs=True,
    )
