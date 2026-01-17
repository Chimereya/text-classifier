"""
Handles loading, saving, and versioning of ML models.
"""

import joblib
import json
import os

MODEL_DIR = "models"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"

# Get the latest model version available
def get_latest_version():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)["latest"]


'''Load the latest model and its vectorizer.
Vectorizer is used to convert text data into numerical features for the model.
Important because the model relies on these features for predictions.'''
def load_model():
    version = get_latest_version()
    if version is None:
        raise RuntimeError("No trained model found.")

    path = f"{MODEL_DIR}/{version}"
    model = joblib.load(f"{path}/model.pkl") # Load the model
    vectorizer = joblib.load(f"{path}/vectorizer.pkl") # Load the vectorizer(to store text features)
    return model, vectorizer, version

def save_model(model, vectorizer, version):
    path = f"{MODEL_DIR}/{version}"
    os.makedirs(path, exist_ok=True) 

    joblib.dump(model, f"{path}/model.pkl")
    joblib.dump(vectorizer, f"{path}/vectorizer.pkl")

    with open(METADATA_PATH, "w") as f:
        json.dump({"latest": version}, f)
