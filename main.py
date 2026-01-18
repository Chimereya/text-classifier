from fastapi import FastAPI, HTTPException
from app.schemas import TextRequest, PredictionResponse
from app.model_manager import load_model
# from train import train(i'll apply this later for train endpoint)

app = FastAPI(title="Production Text Classification API")


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

    return {
        "category": prediction,
        "model_version": version
    }


# @app.post("/train")
# def retrain():
#     version, metrics = train()
#     return {
#         "message": "Model retrained successfully",
#         "model_version": version,
#         "metrics": metrics
#     }
