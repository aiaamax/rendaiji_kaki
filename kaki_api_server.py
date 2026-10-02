"""
kaki_api_server.py
-------------------
Java不要、これ1本で完結するWebアプリ。
- モデルは起動時に1回だけロードして常駐させる
- トップページ(アップロードフォーム)も、判定処理も、このファイルだけでやる

起動方法(ローカル確認用):
    pip install -r requirements.txt
    uvicorn kaki_api_server:app --host 0.0.0.0 --port 8000

Renderの設定:
    Build Command: pip install -r requirements.txt
    Start Command: uvicorn kaki_api_server:app --host 0.0.0.0 --port $PORT
"""

import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# =========================================================
# 設定
# =========================================================
MODEL_PATH = 'best.pt'
INFER_IMG_SIZE = 640  # 推論を軽くするための最大辺サイズ

# 蓮台寺柿特有のお尻判定(下半分)用:オレンジ色のピクセル占有率(%)
HARVEST_THRESHOLDS = {
    'OK_RATIO': 55.0,  # 下半分の55%以上がオレンジなら収穫OK
    'NG_RATIO': 20.0   # 下半分の20%以下ならNG(まだ青い)
}

# =========================================================
# モデルは起動時に1回だけロードする(ここが最重要ポイント)
# =========================================================
print("モデルをロード中...")
model = YOLO(MODEL_PATH)
print("モデルのロード完了。リクエスト受付を開始します。")

app = FastAPI()

# PWA用の静的ファイル(manifest.json, sw.js, アイコン)を配信
# staticフォルダがまだ無い場合はスキップ(サーバーがクラッシュしないように)
import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def get_harvest_label_by_ratio(orange_ratio):
    if orange_ratio >= HARVEST_THRESHOLDS['OK_RATIO']:
        return "OK (Ripe Orange)"
    elif orange_ratio <= HARVEST_THRESHOLDS['NG_RATIO']:
        return "NG (Unripe Green)"
    else:
        return "Check (Yellow)"


def analyze(image_bgr):
    """1枚の画像に対して推論を行い、画面中央に最も近い柿をHSV(下半分面積比率)で分析する"""

    results = model(source=image_bgr, save=False, conf=0.1, iou=0.7,
                     imgsz=INFER_IMG_SIZE, verbose=False)
    result = results[0]

    if result.masks is None or len(result.boxes) == 0:
        return None, "柿が検出されませんでした"

    # --- 画面中央に一番近い柿を1個だけ選別するロジック ---
    img_h, img_w = image_bgr.shape[:2]
    center_x, center_y = img_w / 2, img_h / 2

    best_idx = 0
    min_distance = float('inf')

    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        kaki_center_x = (x1 + x2) / 2
        kaki_center_y = (y1 + y2) / 2
        distance = (kaki_center_x - center_x) ** 2 + (kaki_center_y - center_y) ** 2
        if distance < min_distance:
            min_distance = distance
            best_idx = idx

    # 選ばれた柿のマスクデータを処理
    mask = result.masks.data[best_idx].cpu().numpy()
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    final_mask = (mask_resized > 0.5).astype('uint8') * 255

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "輪郭の検出に失敗しました"

    x, y, w, h = cv2.boundingRect(contours[0])

    # --- 蓮台寺柿対策:柿の下半分(お尻側)だけを分析対象にする ---
    lower_mask = final_mask.copy()
    half_y = y + int(h * 0.5)
    lower_mask[0:half_y, :] = 0

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_values = hsv_image[:, :, 0][lower_mask > 0]

    if len(h_values) == 0:
        return None, "柿の下半分のHSV分析に失敗しました"

    # 下半分の全画素のうち、オレンジ〜黄色(H値が0〜30)のピクセル比率を計算
    total_lower_pixels = len(h_values)
    orange_pixels = np.sum((h_values >= 0) & (h_values <= 30))
    orange_ratio = (orange_pixels / total_lower_pixels) * 100

    harvest_label = get_harvest_label_by_ratio(orange_ratio)
    text_color = (0, 255, 0) if "OK" in harvest_label else (
        (0, 0, 255) if "NG" in harvest_label else (0, 255, 255))
    text_to_draw = f"{harvest_label} (Orange={int(orange_ratio)}%)"

    # 描画処理(お尻部分を薄緑にハイライト)
    annotated = image_bgr.copy()
    mask_color = np.zeros_like(annotated, dtype=np.uint8)
    mask_color[lower_mask > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(annotated, 1, mask_color, 0.4, 0)

    cv2.rectangle(annotated, (x, y), (x + w, y + h), text_color, 4)
    (text_w, text_h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w + 10, y), text_color, -1)
    cv2.putText(annotated, text_to_draw, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated, harvest_label


# =========================================================
# HTML(元のindex.htmlをThymeleafなしのプレーンHTMLに変換したもの)
# =========================================================
def render_page(message: str = "", result_image_data_url: str = ""):
    message_html = f'<div class="message"><p>{message}</p></div>' if message else ""
    result_html = ""
    if result_image_data_url:
        result_html = f'''
        <div>
            <h3>判定結果:</h3>
            <img src="{result_image_data_url}" alt="判定結果画像">
        </div>
        '''

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>柿の収穫判定</title>

    <!-- PWA設定 -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#ff6600">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">

    <style>
        body {{ font-family: sans-serif; text-align: center; background-color: #f4f4f9; margin: 0; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #ff6600; }}
        .upload-btn-wrapper {{ position: relative; overflow: hidden; display: inline-block; margin: 20px 0; }}
        .btn {{ border: 2px solid #ff6600; color: #ff6600; background-color: white; padding: 8px 20px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; }}
        .btn:hover {{ background-color: #ff6600; color: white; }}
        .upload-btn-wrapper input[type=file] {{ font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; }}
        .submit-btn {{ background-color: #ff6600; color: white; border: none; padding: 10px 30px; border-radius: 5px; font-size: 18px; cursor: pointer; margin-top: 10px; }}
        img {{ max-width: 100%; height: auto; margin-top: 20px; border-radius: 5px; border: 1px solid #ddd; }}
        .message {{ margin-top: 10px; font-weight: bold; color: #333; }}
        .loading {{ display: none; color: #666; margin-top: 10px; }}
    </style>
    <script>
        function showLoading() {{
            document.querySelector('.loading').style.display = 'block';
        }}
        if ('serviceWorker' in navigator) {{
            window.addEventListener('load', () => {{
                navigator.serviceWorker.register('/static/sw.js');
            }});
        }}
    </script>
</head>
<body>

<div class="container">
    <h1>柿の収穫判定</h1>
    <p>写真を撮るか、画像を選んでください</p>

    <form method="POST" action="/upload" enctype="multipart/form-data" onsubmit="showLoading()">
        <div class="upload-btn-wrapper">
            <button class="btn" type="button">写真を選択 / 撮影</button>
            <input type="file" name="file" accept="image/*" capture="environment" required>
        </div>
        <br>
        <button type="submit" class="submit-btn">判定開始</button>
    </form>

    <div class="loading">解析中...お待ちください</div>

    {message_html}
    {result_html}
</div>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return render_page()


@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        return render_page(message="画像を選択してください")

    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return render_page(message="画像を読み込めませんでした")

    annotated, label = analyze(image_bgr)

    if annotated is None:
        return render_page(message=f"エラー: {label}")

    ok, buf = cv2.imencode('.jpg', annotated)
    b64_image = base64.b64encode(buf.tobytes()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_image}"

    return render_page(message=f"判定完了！ ({label})", result_image_data_url=data_url)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Flutterアプリ(スマホアプリ)向けのエンドポイント。
    HTMLは返さず、JSON形式で結果だけ返す。
    """
    contents = await file.read()

    if not contents:
        return JSONResponse({"message": "画像を選択してください"}, status_code=400)

    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse({"message": "画像を読み込めませんでした"}, status_code=400)

    annotated, label = analyze(image_bgr)

    if annotated is None:
        return JSONResponse({"message": f"エラー: {label}"}, status_code=200)

    ok, buf = cv2.imencode('.jpg', annotated)
    b64_image = base64.b64encode(buf.tobytes()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_image}"

    return JSONResponse({
        "message": f"判定完了！ ({label})",
        "label": label,
        "resultImage": data_url
    })
