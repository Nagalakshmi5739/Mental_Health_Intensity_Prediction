# Mental Health Intensity Prediction

**Production-ready Flask application** for real-time mental health text sentiment classification using a GRU neural network trained on GloVe embeddings.

---

## Project Overview

This project deploys a trained NLP model as a web service. Users input text, and the model predicts the emotional intensity category between four classes: **Negative**, **Neutral**, **Positive**, and **Very Negative**, along with confidence scores.

- **Model:** GRU (Gated Recurrent Unit)
- **Embeddings:** GloVe 100-dimensional word vectors
- **Framework:** TensorFlow / Keras
- **Frontend:** Responsive HTML/CSS/JavaScript with glassmorphism UI
- **Backend:** Flask + Gunicorn (production)

---

## Features

- Real-time text prediction with confidence scores
- Preprocessing pipeline identical to training
- Secure production deployment with Gunicorn
- Health check endpoint for monitoring
- Clean, portfolio-ready UI with loading states
- Graceful error handling and logging

---

## Folder Structure

```
Mental_Health_Flask/
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
├── README.md
├── models/
│   ├── glove_gru.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
├── utils/
│   ├── preprocess.py
│   └── predictor.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── uploads/
```

---

## Prerequisites

- Python >= 3.8, <= 3.11 (TensorFlow compatibility)
- pip
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd Mental_Health_Flask
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure model files are present

Move the following files into the `models/` directory:
- `glove_gru.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`

If you need to generate these, run the training notebook first using:
```python
# Save model
model.save("glove_gru.keras")

# Save tokenizer
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
```

---

## Batch Upload

You can upload a **CSV or Excel file** to run predictions on multiple texts at once.

The uploaded file must contain:
- A **text column** named `posts`, `text`, `content`, or any column with string values longer than 5 characters.
- A **label column** named `predicted`, `actual`, `label`, `intensity`, `target`, `class`, or `sentiment`.

### Usage

1. Click the **Choose a CSV or Excel file** button in the Batch Upload section.
2. Select your file (`.csv`, `.xlsx`, `.xls`).
3. Click **Predict All**.
4. Results appear below with predicted labels, actual labels, and correctness per row.
5. Use **Download CSV** to get a file with all results.

### Example CSV format

```csv
posts,predicted
I feel great today!,positive
I am so sad and lonely.,negative
The weather is okay.,neutral
```

---

## Running Locally

```bash
set FLASK_APP=app.py
flask run
```

Or simply:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Local Testing

### Verify model loading
```bash
python -c "from app import model, tokenizer, label_encoder; print('OK')"
```

### Verify predictions
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{\"text_input\": \"I feel anxious and overwhelmed\"}'
```

### Health check
```bash
curl http://127.0.0.1:5000/health
```

---

## Deployment

### Render

1. Push code to GitHub.
2. Create a new **Web Service** on Render.
3. Connect your repository.
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --log-file -`
   - **Runtime:** Python 3
5. Deploy.

### Railway

1. Push code to GitHub.
2. Create a new **Project** on Railway.
3. Connect your repository.
4. Railway auto-detects the Procfile and deploys.
5. Default port is set by environment variable `PORT`.

### PythonAnywhere

1. Upload the project files.
2. Open a Bash console and create a virtualenv:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Go to **Web** tab -> **Manual configuration**.
4. Set working directory to `/home/<YOUR_USER>/Mental_Health_Flask`.
5. Set **WSGI configuration** to point to `app.py`.
6. Reload the web app.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port the server binds to |
| `FLASK_APP` | `app.py` | Flask entry point |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **Model file not found** | Ensure `glove_gru.keras`, `tokenizer.pkl`, and `label_encoder.pkl` are inside the `models/` folder. |
| **TemplateNotFound** | Ensure `templates/index.html` exists and Flask is run from the project root. |
| **ModuleNotFoundError** | Activate the virtual environment and run `pip install -r requirements.txt`. |
| **TensorFlow version mismatch** | Use Python 3.8-3.11 and TensorFlow 2.15.0 for best compatibility. |
| **Protobuf error** | Pin `protobuf==3.20.3` in requirements. |
| **Out of memory** | Reduce batch size during training or use a smaller model. |
| **Port already in use** | Change port with `python app.py` or set `PORT` env variable. |

---

## Future Improvements

- Add model explainability (e.g., SHAP or LIME)
- Batch prediction API
- User authentication and history
- Docker containerization
- Model versioning endpoint
- Frontend unit tests with Jest/Cypress
- CI/CD pipeline with GitHub Actions

---

## License

MIT License - feel free to use this project for learning and portfolio purposes.
