"""
kaki_api_server.py
-------------------
モデルを起動時に1回だけロードして常駐させるFastAPIサーバー。
Java側はプロセスを毎回起動する代わりに、このサーバーにHTTPで画像を投げるだけになる。

起動方法:
    pip install fastapi uvicorn python-multipart ultralytics opencv-python-headless
    uvicorn kaki_api_server:app --host 0.0.0.0 --port 8000

Renderにデプロイする場合は、このスクリプトを"Web Service"としてデプロイし、
Start Commandを下記のようにする:
    uvicorn kaki_api_server:app --host 0.0.0.0 --port $PORT
"""

import base64
import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================
MODEL_PATH = 'best.pt'

# 推論を軽くするための最大辺サイズ(必要に応じて調整)
INFER_IMG_SIZE = 640

HARVEST_THRESHOLDS = {
    'OK_ORANGE': 25,
    'NG_GREEN': 40
}

# =========================================================
# モデルはここで「起動時に1回だけ」ロードする
# これがJavaのProcessBuilder方式との一番の違い
# =========================================================
print("🚀 モデルをロード中...(サーバー起動時の1回だけ)")
model = YOLO(MODEL_PATH)
print("✅ モデルのロード完了。リクエスト受付を開始します。")

app = FastAPI()


def get_harvest_label(mean_h):
    if mean_h < HARVEST_THRESHOLDS['OK_ORANGE']:
        return "OK (Ripe Orange)", (0, 255, 0)
    elif mean_h >= HARVEST_THRESHOLDS['NG_GREEN']:
        return "NG (Unripe Green)", (0, 0, 255)
    else:
        return "Check (Yellow)", (0, 255, 255)


def analyze(image_bgr):
    """1枚の画像に対して推論・HSV分析・描画を行い、結果を返す"""

    results = model(source=image_bgr, save=False, conf=0.1, iou=0.7,
                     imgsz=INFER_IMG_SIZE, verbose=False)
    result = results[0]

    if result.masks is None:
        return None, "No Kaki Detected"

    mask = result.masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    final_mask = (mask_resized > 0.5).astype('uint8') * 255

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "Mask Error"

    x, y, w, h = cv2.boundingRect(contours[0])

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_values = hsv_image[:, :, 0][final_mask > 0]

    if len(h_values) == 0:
        return None, "HSV Analysis Failed"

    mean_h = int(np.mean(h_values))
    harvest_label, text_color = get_harvest_label(mean_h)
    text_to_draw = f"{harvest_label} (H={mean_h})"

    mask_color = np.zeros_like(image_bgr, dtype=np.uint8)
    mask_color[final_mask > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(image_bgr, 1, mask_color, 0.5, 0)

    cv2.rectangle(annotated, (x, y), (x + w, y + h), text_color, 4)
    (text_w, text_h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w + 10, y), text_color, -1)
    cv2.putText(annotated, text_to_draw, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated, harvest_label


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse({"message": "画像を読み込めませんでした"}, status_code=400)

    annotated, label = analyze(image_bgr)

    if annotated is None:
        return JSONResponse({"message": f"判定不可: {label}"}, status_code=200)

    ok, buf = cv2.imencode('.jpg', annotated)
    b64_image = base64.b64encode(buf.tobytes()).decode('utf-8')

    return JSONResponse({
        "message": "判定完了！",
        "label": label,
        "resultImage": f"data:image/jpeg;base64,{b64_image}"
    })


@app.get("/health")
async def health():
    # Render等のヘルスチェック用。モデルが読み込み済みか確認できる
    return {"status": "ok", "model_loaded": model is not None}
