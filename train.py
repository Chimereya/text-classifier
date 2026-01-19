''' offline training script. for now were using CLI for the training. 
later i'll apply /train endpoint'''


import pandas as pd
import uuid 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.model_manager import save_model
from app.metrics import generate_metrics


DATA_PATH = "data/tickets.csv"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # im using 20% for testing
    stratify=y,      
    random_state=42   # Sets seed for reproducibility
)
    

    vectorizer = TfidfVectorizer(
        max_features=5000, # limit to top 5000 features
        ngram_range=(1, 2), 
        stop_words='english' # remove common english stop words to improve performance
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test) 

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    metrics = generate_metrics(y_test, preds) 

    version = f"v_{uuid.uuid4()}.hex[:8]" # unique version identifier for each model
    save_model(model, vectorizer, version)

    print(f"Model trained and saved with version: {version}")
    return version, metrics

if __name__ == "__main__":
    v, m = train()
    print(f"Trained Model Version: {v}")
    print("Metrics:", m)