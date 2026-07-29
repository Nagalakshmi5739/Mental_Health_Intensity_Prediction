import os
import warnings
import logging
import traceback
import pickle


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from keras.models import load_model
import torch
torch.set_num_threads(1)

from utils.predictor import predict_mental_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "glove_gru.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


model = None
tokenizer = None
label_encoder = None


def optimize_tensorflow():
    pass


def warmup_model(model_local, tokenizer_local):
    """Run a dummy inference to warm up the model."""
    try:
        dummy_text = "warmup"
        dummy_padded = tokenizer_local.texts_to_sequences([dummy_text])
        from keras.utils import pad_sequences
        dummy_padded = pad_sequences(
            dummy_padded, maxlen=100, padding="pre", truncating="pre"
        )
        import torch
        with torch.no_grad():
            model_local(dummy_padded, training=False)
        logger.info("Model warmup completed successfully.")
    except Exception as exc:
        logger.warning("Model warmup failed: %s", exc)


def load_model_artifacts():
    """
    Load the trained GRU model, tokenizer, and label encoder.

    Returns:
        tuple: (model, tokenizer, label_encoder)

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    pass

    for path, name in [
        (MODEL_PATH, "Keras model"),
        (TOKENIZER_PATH, "Tokenizer"),
        (LABEL_ENCODER_PATH, "Label encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at {path}. "
                "Ensure the trained model files are saved in the 'models/' directory."
            )

    logger.info("Loading model artifacts from: %s", MODELS_DIR)
    model_local = load_model(MODEL_PATH)
    logger.info("Model loaded successfully. Input shape: %s", model_local.input_shape)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer_local = pickle.load(f)
    logger.info("Tokenizer loaded successfully. Vocabulary size: %s", tokenizer_local.num_words)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder_local = pickle.load(f)
    logger.info(
        "Label encoder loaded successfully. Classes: %s",
        [int(c) for c in label_encoder_local.classes_],
    )

    warmup_model(model_local, tokenizer_local)

    return model_local, tokenizer_local, label_encoder_local


try:
    model, tokenizer, label_encoder = load_model_artifacts()
    logger.info("All artifacts loaded successfully. Server is ready.")
except Exception as exc:
    logger.critical("Failed to load model artifacts: %s", exc)
    logger.critical("Server will start but predictions will fail until this is fixed.")
    model = None
    tokenizer = None
    label_encoder = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to receive user text and return prediction results as JSON.
    Expected JSON field: 'text_input'
    """
    try:
        data = request.get_json(force=True)
        text = data.get("text_input", "").strip()

        if not text:
            logger.warning("Prediction request with empty text.")
            return jsonify(
                {
                    "error": "Please enter some text before predicting.",
                    "label": None,
                    "confidence": None,
                }
            ), 400

        if model is None or tokenizer is None or label_encoder is None:
            logger.error("Model artifacts not loaded. Cannot make predictions.")
            return jsonify(
                {
                    "error": "Model is not loaded. Please check server logs.",
                    "label": None,
                    "confidence": None,
                }
            ), 500

        logger.info("Running prediction for text: %s", text[:100])
        result = predict_mental_health(text, model, tokenizer, label_encoder)
        logger.info(
            "Prediction successful for input length %d. Label: %s, Confidence: %.2f%%",
            len(text),
            result["label"],
            result["confidence"],
        )
        return jsonify(result)

    except Exception as exc:
        logger.error("Error during prediction: %s", traceback.format_exc())
        return jsonify(
            {
                "error": f"An internal error occurred: {str(exc)}",
                "label": None,
                "confidence": None,
            }
        ), 500


@app.route("/test-predict", methods=["POST"])
def test_predict():
    """Debug endpoint that returns a dummy prediction without using the model."""
    try:
        data = request.get_json(force=True)
        text = data.get("text_input", "").strip() or "test"
        return jsonify({
            "label": "neutral",
            "confidence": 50.0,
            "all_probabilities": {
                "very negative": 10.0,
                "negative": 20.0,
                "neutral": 50.0,
                "positive": 20.0,
            },
            "debug": "dummy prediction - model not called",
            "text_received": text[:100],
        })
    except Exception as exc:
        logger.error("Error in test-predict: %s", traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg", ".ico"
}


def _allowed_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


def _is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def _detect_text_column(df):
    candidates = ["posts", "text", "content", "post", "comment", "message"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "string":
            sample = df[col].dropna().head(20).tolist()
            if sample and isinstance(sample[0], str) and len(str(sample[0])) > 5:
                return col
    raise ValueError(
        "No text column found. Expected one of: "
        + ", ".join(candidates)
        + " or any column with string values longer than 5 characters."
    )


def _detect_actual_column(df):
    candidates = ["predicted", "actual", "label", "intensity", "target", "class", "sentiment"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No actual/label column found. Expected one of: "
        + ", ".join(candidates)
        + "."
    )


@app.route("/upload", methods=["POST"])
def upload_predict():
    import pandas as pd
    """
    Endpoint to upload a CSV or Excel file for batch prediction.
    Expected file field: 'file'
    Expected columns: a text column (posts/text/content/etc.) and
                       a label column (predicted/actual/label/etc.)
    Returns a CSV download with predicted values, actual values, and correctness.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use 'file' field."}), 400

        file = request.files["file"]
        safe_name = secure_filename(file.filename or "")
        if not safe_name or safe_name.strip() == "":
            return jsonify({"error": "No file selected or invalid filename."}), 400

        if _is_image_file(safe_name):
            return jsonify(
                {
                    "error": f"Cannot read '{safe_name}' (this model does not support image input). Please upload a CSV or Excel file instead."
                }
            ), 400

        if not _allowed_file(safe_name):
            return jsonify(
                {
                    "error": "Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, .xls)."
                }
            ), 400

        ext = os.path.splitext(safe_name)[1].lower()
        filename = f"batch_{os.getpid()}_{safe_name}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        if ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        if df.empty:
            os.remove(filepath)
            return jsonify({"error": "The uploaded file is empty."}), 400

        text_col = _detect_text_column(df)
        actual_col = _detect_actual_column(df)

        results = []
        correct_count = 0

        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            actual_label = str(row[actual_col]) if pd.notna(row[actual_col]) else ""

            if not raw_text.strip():
                results.append(
                    {
                        "text": raw_text,
                        "predicted": "",
                        "actual": actual_label,
                        "correct": False,
                    }
                )
                continue

            pred_result = predict_mental_health(raw_text, model, tokenizer, label_encoder)
            predicted_label = pred_result["label"]
            is_correct = predicted_label.lower() == actual_label.strip().lower()
            if is_correct:
                correct_count += 1
            results.append(
                {
                    "text": raw_text[:200],
                    "predicted": predicted_label,
                    "actual": actual_label,
                    "correct": is_correct,
                }
            )

        os.remove(filepath)

        output_df = pd.DataFrame(results)
        csv_path = os.path.join(UPLOAD_DIR, f"batch_result_{os.getpid()}.csv")
        output_df.to_csv(csv_path, index=False)

        total = len(results)
        logger.info(
            "Batch prediction complete: %d rows, %d correct (%.1f%%)",
            total,
            correct_count,
            (correct_count / total * 100) if total > 0 else 0,
        )

        return send_file(
            csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"batch_results_{os.getpid()}.csv",
        )

    except Exception as exc:
        logger.error("Error during batch prediction: %s", traceback.format_exc())
        return jsonify(
            {
                "error": f"Batch prediction failed: {str(exc)}",
            }
        ), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for deployment platforms.
    """
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "label_encoder_loaded": label_encoder is not None,
        "models_dir": MODELS_DIR,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False,
        "tokenizer_exists": os.path.exists(TOKENIZER_PATH) if TOKENIZER_PATH else False,
        "label_encoder_exists": os.path.exists(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH else False,
    }
    return jsonify(status)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)