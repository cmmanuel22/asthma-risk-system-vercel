import os
class FusionOutput:
final_risk: str
risk_score: float
confidence: float
reasoning: str
individual_risks: Dict[str, int]
spo2_was_critical: bool


# Minimal feature extraction
def create_mel_spectrogram(file_stream) -> np.ndarray:
try:
signal, sr = librosa.load(file_stream, sr=SAMPLE_RATE)
max_len_samples = sr * MAX_CLIP_SECONDS
if len(signal) > max_len_samples:
signal = signal[:max_len_samples]
elif len(signal) < max_len_samples:
signal = np.pad(signal, (0, max_len_samples - len(signal)), mode='constant')


mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
if max_val == min_val:
return np.zeros_like(mel_spec_db)[..., np.newaxis]
norm_spec = (mel_spec_db - min_val) / (max_val - min_val)
return norm_spec[..., np.newaxis]
except Exception as e:
print(f"Error extracting features: {e}")
return None


@app.route('/api/predict', methods=['POST'])
def predict():
if interpreter is None:
return jsonify({"error": "Model/interpreter not available on server."}), 500


if 'audio_file' not in request.files or 'spo2' not in request.form or 'bpm' not in request.form:
return jsonify({"error": "Missing form data: audio_file, spo2, bpm are required."}), 400


audio_file = request.files['audio_file']
try:
spo2_value = float(request.form['spo2'])
bpm_value = float(request.form['bpm'])
except ValueError:
return jsonify({"error": "spo2 and bpm must be numeric."}), 400


feature = create_mel_spectrogram(audio_file.stream)
if feature is None:
return jsonify({"error": "Failed to extract audio features."}), 500


input_data = np.expand_dims(feature, axis=0).astype(np.float32)
try:
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
audio_risk = int(np.argmax(output_data))
except Exception as e:
return jsonify({"error": f"Model inference failed: {e}"}), 500


# placeholder fusion output (the same structure you already have)
fusion = FusionOutput(final_risk="MEDIUM", risk_score=1.0, confidence=0.9, reasoning="demo", individual_risks={"audio": audio_risk, "spo2": 1, "breathing": 1}, spo2_was_critical=False)
return jsonify(asdict(fusion))


# Vercel expects an `app` object exported
app = app