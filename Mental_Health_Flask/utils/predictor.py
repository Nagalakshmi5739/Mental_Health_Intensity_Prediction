"""
Prediction module for mental health text classification.

Wraps model inference, tokenization, padding, and label decoding
so the Flask app stays clean and focused on routing.
"""
import numpy as np
from keras.utils import pad_sequences

from utils.preprocess import preprocess_text

MAX_SEQUENCE_LENGTH = 100

LABEL_MAP = {
    -2: "very negative",
    -1: "negative",
    0: "neutral",
    1: "positive",
}


def tokenize_and_pad(text: str, tokenizer):
    """
    Convert preprocessed text into padded integer sequences.

    Args:
        text: Preprocessed text string from preprocess_text().
        tokenizer: Loaded Keras Tokenizer instance.

    Returns:
        np.ndarray: Padded sequence of shape (1, MAX_SEQUENCE_LENGTH).
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="pre",
        truncating="pre",
    )
    return padded


def predict_mental_health(text: str, model, tokenizer, label_encoder) -> dict:
    """
    Run inference on the given text.

    Pipeline:
    1. Preprocess raw text
    2. Tokenize
    3. Pad sequence
    4. Model prediction (softmax output)
    5. Decode predicted class via label encoder

    Args:
        text: Raw user input string.
        model: Loaded Keras model.
        tokenizer: Loaded Keras Tokenizer instance.
        label_encoder: Loaded sklearn LabelEncoder instance.

    Returns:
        dict: Contains 'label', 'confidence', and 'all_probabilities'.
    """
    cleaned = preprocess_text(text)
    padded = tokenize_and_pad(cleaned, tokenizer)
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    raw_label = label_encoder.inverse_transform([predicted_index])[0]
    predicted_label = LABEL_MAP.get(int(raw_label), str(raw_label))
    confidence = float(probabilities[predicted_index])
    all_classes = [LABEL_MAP.get(int(c), str(c)) for c in label_encoder.classes_]

    return {
        "label": predicted_label,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            class_name: round(float(prob) * 100, 2)
            for class_name, prob in zip(all_classes, probabilities)
        },
    }