import io, json
import torch, torch.nn as nn
from torchvision import models, transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

META = json.load(open("model/meta.json"))
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/best_fp16.pt", map_location="cpu"))
model = model.float().eval()

tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

app = FastAPI(title="Pneumonia CXR Classifier",
              description="Demo penelitian. BUKAN alat diagnosis medis.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Unggah JPEG atau PNG.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Maksimal 10 MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar valid.")
    with torch.inference_mode():
        p = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0, 1].item()
    return {"probability_pneumonia": round(p, 4),
            "label": META["classes"][int(p >= META["threshold"])],
            "threshold": META["threshold"],
            "disclaimer": "Demo penelitian, bukan alat diagnosis."}