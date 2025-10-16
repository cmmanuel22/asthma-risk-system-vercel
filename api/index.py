import os
from flask import Flask, request, jsonify
import librosa
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict

# --- TFLite Runtime Compatibility (Corrected) ---
# CRITICAL FIX: The interpreter is often available directly under the tflite_runtime module,
# not necessarily inside an 'interpreter' submodule like in full TensorFlow.
try:
    # Attempt to import Interpreter directly from tflite_runtime (common pattern)
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime.interpreter.Interpreter")
except ImportError:
    try:
        # Some build environments place it right at the top level
        from tflite_runtime import Interpreter
        print("Using tflite_runtime.Interpreter (Top Level)")
    except ImportError:
        # Fallback to full TensorFlow for local testing/development
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (Full TF fallback)")


# --- Configuration ---
# Vercel's Serverless Function should be able to find a small file 
# placed in the same directory as the function file (api/index.py).
# Assuming 'audio_model_standard.tflite' is in the 'api' directory.
MODEL_FILENAME = "audio_model_standard.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_CLIP_SECONDS = 3
AUDIO_CATEGORIES = {0: "SAFE", 1: "MEDIUM", 2: "HIGH"}

# --- Initialize Flask App and Load Model ---
app = Flask(__name__)

# Load the model only once when the function starts
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    # This exception now handles both file-not-found AND library-not-found issues
    print(f"❌ FATAL ERROR: Could not load TFLite model or initialize Interpreter: {e}")
    interpreter = None

# --- Risk Fusion Logic (Remaining Code - Unchanged) ---
SENSOR_WEIGHTS: Dict[str, float] = {"audio": 1.0, "spo2": 2.5, "breathing": 1.5}
TOTAL_WEIGHT: float = sum(SENSOR_WEIGHTS.values())
RISK_THRESHOLDS: Dict[str, float] = {"safe_max": 0.67, "medium_max": 1.33, "high_min": 1.33}
SPO2_THRESHOLDS: Dict[str, int] = {"high_max": 92, "safe_min": 95}
BREATHING_THRESHOLDS_3_TO_7_YRS: Dict[str, int] = {"safe_max": 34, "medium_max": 40}

@dataclass
class FusionOutput:
    final_risk: str
    risk_score: float
    confidence: float
    reasoning: str
    individual_risks: Dict[str, int]
    spo2_was_critical: bool

def classify_spo2(spo2_value: float) -> int:
    if spo2_value <= SPO2_THRESHOLDS["high_max"]: return 2
    elif spo2_value < SPO2_THRESHOLDS["safe_min"]: return 1
    else: return 0

def classify_breathing_rate(bpm: float) -> int:
    if bpm > BREATHING_THRESHOLDS_3_TO_7_YRS["medium_max"]: return 2
    elif bpm > BREATHING_THRESHOLDS_3_TO_7_YRS["safe_max"]: return 1
    else: return 0

def hybrid_fusion(audio_risk: int, spo2_value: float, bpm: float) -> FusionOutput:
    spo2_risk = classify_spo2(spo2_value)
    breathing_risk = classify_breathing_rate(bpm)
    individual_risks = {"audio": audio_risk, "spo2": spo2_risk, "breathing": breathing_risk}

    if spo2_risk == 2:
        reasoning = "CRITICAL OVERRIDE: SpO2 at or below 92% triggered the safety guardrail."
        return FusionOutput("HIGH", 2.0, 0.95, reasoning, individual_risks, True)

    weighted_sum = (SENSOR_WEIGHTS["audio"] * audio_risk + SENSOR_WEIGHTS["spo2"] * spo2_risk + SENSOR_WEIGHTS["breathing"] * breathing_risk)
    risk_score = weighted_sum / TOTAL_WEIGHT
    
    final_risk = "HIGH" if risk_score >= RISK_THRESHOLDS["high_min"] else "MEDIUM" if risk_score > RISK_THRESHOLDS["safe_max"] else "SAFE"
    confidence = max(0.5, 1.0 - (np.std(list(individual_risks.values())) * 0.5))
    reasoning = f"Weighted fusion score of {risk_score:.2f} resulted in a {final_risk} risk assessment."
    return FusionOutput(final_risk, risk_score, confidence, reasoning, individual_risks, False)

# --- API-Specific Functions (Unchanged logic) ---
def create_mel_spectrogram(file_stream) -> np.ndarray:
    try:
        signal, sr = librosa.load(file_stream, sr=SAMPLE_RATE)
        max_len_samples = sr * MAX_CLIP_SECONDS
        if len(signal) > max_len_samples: signal = signal[:max_len_samples]
        elif len(signal) < max_len_samples: signal = np.pad(signal, (0, max_len_samples - len(signal)), mode='constant')
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
        if max_val == min_val: return np.zeros_like(mel_spec_db)[..., np.newaxis]
        norm_spec = (mel_spec_db - min_val) / (max_val - min_val)
        return norm_spec[..., np.newaxis]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route("/api/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model is not loaded on the server. Check build logs for dependency errors."}), 500

    if 'audio_file' not in request.files or 'spo2' not in request.form or 'bpm' not in request.form:
        return jsonify({"error": "Missing required form data: audio_file, spo2, and bpm."}), 400

    audio_file = request.files['audio_file']
    
    try:
        spo2_value = float(request.form['spo2'])
        bpm_value = float(request.form['bpm'])
    except ValueError:
        return jsonify({"error": "Invalid input: spo2 and bpm must be numbers."}), 400

    feature = create_mel_spectrogram(audio_file.stream)
    if feature is None:
        return jsonify({"error": "Failed to extract audio features."}), 500
        
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    audio_risk = int(np.argmax(output_data))
    
    fusion_result = hybrid_fusion(audio_risk, spo2_value, bpm_value)

    return jsonify(asdict(fusion_result))

# This is the entry point for Vercel
app = app