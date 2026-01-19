# Text Classifier

A simple machine learning model that analyzes the text of a movie review from the IMDb dataset to automatically categorize its sentiment, usually as either positive or negative.

---

## App Preview

![App Screenshot](screenshot.png) coming soon
*Above: The interactive Swagger UI showing the available API endpoints.*

---

## Key Features

* **Text preprocessing:**.
* **TF-IDF → Logistic Regression model:**.


## Dataset 
Check out the [Dataset](https://tinyurl.com/bddmvv9j) for more information.


---

## Tech Stack

* **Language:** Python 3.10+
* **Framework:** FastAPI
* **Web Server:** Uvicorn

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Chimereya/text-classifier.git
cd your-repo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# download and add the IMDB dataset to the data/ directory