"""
Prediction module for mental health text classification.

Wraps model inference, tokenization, padding, and label decoding
so the Flask app stays clean and focused on routing.
"""
import numpy as np

from utils.preprocess import preprocess_text

MAX_SEQUENCE_LENGTH = 100

LABEL_MAP = {
    -2: "very negative",
    -1: "negative",
    0: "neutral",
    1: "positive",
}


def tokenize_and_pad(text: str, word_index: dict):
    """
    Convert preprocessed text into padded integer sequences manually.

    Args:
        text: Preprocessed text string from preprocess_text().
        word_index: Dictionary mapping words to integer indices.

    Returns:
        np.ndarray: Padded sequence of shape (1, MAX_SEQUENCE_LENGTH).
    """
    tokens = text.split()
    # Keras Tokenizer converts out-of-vocabulary words to nothing (skips them) by default.
    # Words in word_index start at 1.
    sequence = []
    for word in tokens:
        idx = word_index.get(word)
        if idx is not None:
            sequence.append(idx)
            
    # Pad or truncate (pre-padding and pre-truncating)
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        sequence = sequence[-MAX_SEQUENCE_LENGTH:]
    else:
        sequence = [0] * (MAX_SEQUENCE_LENGTH - len(sequence)) + sequence

    return np.array([sequence], dtype=np.float32)


def predict_mental_health(text: str, interpreter, word_index: dict, label_encoder) -> dict:
    """
    Run inference on the given text using TFLite.

    Pipeline:
    1. Preprocess raw text
    2. Tokenize and pad sequence using word_index
    3. Model prediction (TFLite interpreter)
    4. Decode predicted class via label encoder

    Args:
        text: Raw user input string.
        interpreter: Loaded tflite_runtime Interpreter.
        word_index: Loaded dictionary mapping word -> id.
        label_encoder: Loaded sklearn LabelEncoder instance.

    Returns:
        dict: Contains 'label', 'confidence', and 'all_probabilities'.
    """
    cleaned = preprocess_text(text)
    padded = tokenize_and_pad(cleaned, word_index)

    # TFLite inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]

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