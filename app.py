from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import requests
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

PKL_URL = "https://raw.githubusercontent.com/Yeongrok93/UlcerPredcition/main/model/ulcer_prediction.pkl"
PKL_FILE = "ulcer_prediction.pkl"

def download_and_load_model():
    """항상 GitHub에서 최신 pkl 받아오기 (디스크 캐시 문제 방지)."""
    print("⬇️ Downloading latest model from GitHub...")
    response = requests.get(PKL_URL, timeout=30)
    response.raise_for_status()
    with open(PKL_FILE, "wb") as f:
        f.write(response.content)
    print(f"✅ Downloaded: {os.path.getsize(PKL_FILE)} bytes")
    return joblib.load(PKL_FILE)


model = download_and_load_model()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rass = float(request.form['feature1'])
        strength = float(request.form['feature2'])
        tmax = float(request.form['feature3'])
        tmin = float(request.form['feature4'])
        incontinence = float(request.form['feature5'])

        if not (-5 <= rass <= 4):
            return jsonify({"error": "Level of consciousness (RASS) must be between -5 and +4."})
        if not (0 <= strength <= 10):
            return jsonify({"error": "Lower extremity muscle strength must be between 0 and 10."})

        new_data = pd.DataFrame([[rass, strength, tmax, tmin, incontinence]],
                                columns=['RASS 평균값', 'motor strength LE 평균', '체온(최대)', '체온(최소)', '하루평균실금'])

        probability = model.predict_proba(new_data)[0, 1] * 100

        return jsonify({"probability": round(probability, 1)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
