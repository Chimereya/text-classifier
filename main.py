from fastapi import FastAPI, HTTPException
from app.schemas import TextRequest, PredictionResponse
from app.model_manager import load_model
from fastapi.middleware.cors import CORSMiddleware

from train import train

app = FastAPI(title="Production Text Classification API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # we'll allow this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return{"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    try:
        model, vectorizer, version = load_model()  # Load the latest model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    text_vec = vectorizer.transform([request.text])
    prediction = model.predict(text_vec)[0]

    # Confidence
    probabilities = model.predict_proba(text_vec)[0]
    confidence = float(max(probabilities))


    return {
        "category": prediction,
        "model_version": version,
        "confidence": round(confidence, 3),
    }


@app.post("/train")
def retrain():
    version, metrics = train()
    return {
        "message": "Model retrained successfully",
        "model_version": version,
        "metrics": metrics
    }
