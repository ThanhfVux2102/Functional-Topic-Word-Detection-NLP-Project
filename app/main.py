from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from typing import Optional

# config endpoint
MODEL_PATH = Path("models/text_classifier.pkl")



# load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)



app = FastAPI(
    title="Text Classifier API",
    description="API phục vụ model phân loại văn bản ",
    version="1.0.0",
)


# schema
class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    prediction: str
    probability: Optional[float] = None  



@app.get("/")
def root():
    return {"message": "Text classifier API is running."}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    """
    Nhận 1 câu text, trả về nhãn dự đoán (prediction)
    và probability nếu model hỗ trợ predict_proba.
    """
    text = payload.text

    y_pred = model.predict([text])[0]

# predict_proba if model can
    proba = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba([text])[0]
        proba = float(probas.max())

    return PredictionOut(
        prediction=str(y_pred),
        probability=proba,
    )
