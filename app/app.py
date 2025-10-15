from flask import Flask, request, render_template, jsonify
import openai
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# ------------------ Load Models ------------------
# Main Parkinson’s detection model (binary)
parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

# Severity prediction model (multi-class)
parkinsons_severity = joblib.load("Parkinsons_severity.pkl")

# Label encoder used when training the severity model
label_encoder = joblib.load("label_encoder.pkl")


# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("predict.html")

# ------------------ Load Models ------------------
# Individual models
models = {
    'XGBoost': pickle.load(open("xgb_model.pkl", "rb")),
    'SVM': pickle.load(open("svm_model.pkl", "rb")),
    'LogisticRegression': pickle.load(open("lr_model.pkl", "rb"))
}
# Ensemble
ensemble_model = pickle.load(open("ensemble_model.pkl", "rb"))
# Scaler used during training
scaler = joblib.load("scaler.pkl")
# ------------------ Feature columns ------------------
feature_columns = ['MDVPFoHz','MDVPFhiHz','MDVPFloHz','MDVPJitter','MDVPShimmer',
                   'HNR','RPDE','DFA','spread1','spread2','D2']
# ------------------ Predict Route ------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data received"}), 400

    # Extract 11 features in correct order
    features = [float(data.get(key, 0.0)) for key in ['fo','fhi','flo','Jitter','Shimmer',
                                                     'HNR','RPDE','DFA','Spread1','Spread2','D2']]
    
    # Convert to 2D array for sklearn
    features_array = np.array([features], dtype=float)
    
    # Scale features
    features_scaled = scaler.transform(features_array)

    # ---------------- Individual Model Predictions ----------------
    predictions = {}
    for name, model in models.items():
        prob = model.predict_proba(features_scaled)[0][1]  # probability of PD
        label = "Parkinson's Disease" if prob >= 0.5 else "Healthy"
        predictions[name] = {"label": label, "prob": float(round(prob, 3))}  # convert to Python float

    # ---------------- Ensemble Prediction ----------------
    ensemble_prob = ensemble_model.predict_proba(features_scaled)[0][1]
    ensemble_label = "Parkinson's Disease" if ensemble_prob >= 0.5 else "Healthy"
    predictions['Ensemble'] = {"label": ensemble_label, "prob": float(round(ensemble_prob, 3))}

    # Convert features to Python floats
    features_python = [float(f) for f in features]

    return jsonify({"predictions": predictions, "features": features_python})


# ------------------ Parkinson's Severity Prediction ------------------
@app.route("/predict_severity", methods=["POST"])
def predict_severity():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data"}), 400

    # Extract features sent from HTML
    features = data.get("features", [])

    # Convert to float and trim to 21 features
    features = [float(x) for x in features][:21]

    # Safety check
    if len(features) != 21:
        return jsonify({"error": f"Expected 21 features, got {len(features)}"}), 400

    # Predict
    severity_pred = parkinsons_severity.predict([features])[0]
    severity_label = label_encoder.inverse_transform([severity_pred])[0]

    # Provide precautions
    precautions = {
        "Mild": [
            "Continue regular physical activity (yoga, stretching, walking).",
            "Follow a balanced diet rich in antioxidants.",
            "Schedule neurologist visits every 6 months.",
            "Stay socially and mentally active."
        ],
        "Moderate": [
            "Maintain consistent medication routines.",
            "Engage in physiotherapy for mobility and coordination.",
            "Avoid fatigue; get adequate rest.",
            "Ensure home safety (anti-slip floors, railings)."
        ],
        "Severe": [
            "Use assistive tools for walking and eating.",
            "Have caregiver assistance during daily tasks.",
            "Frequent neurological monitoring (monthly if advised).",
            "Stick to prescribed therapy plans."
        ]
    }

    return jsonify({
        "severity": severity_label,
        "precautions": precautions.get(severity_label, [])
    })



# ------------------ AI Advice Generation (Optional) ------------------
@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    """
    Generates AI-based or fallback advice depending on prediction result.
    """
    data = request.get_json()
    prediction = data.get('prediction', '')

    if 'Parkinson' in prediction:
        try:
            # ⚠️ IMPORTANT: never hardcode keys in production!
           
            import os
            openai.api_key = os.getenv("OPENAI_API_KEY")

            prompt = (
                "Provide safe, clear, and empathetic medical self-care tips for a person "
                "with early signs of Parkinson's disease. Include lifestyle, exercise, and safety advice "
                "in 5 short bullet points."
            )

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            advice_text = response.choices[0].message['content']

        except Exception:
            # Fallback if OpenAI API fails
            advice_text = (
                "<ul>"
                "<li>Consult a neurologist for early diagnosis and treatment.</li>"
                "<li>Engage in light physical exercises like walking or yoga.</li>"
                "<li>Maintain a balanced diet rich in fiber and hydration.</li>"
                "<li>Ensure a safe home setup — remove fall hazards and add railings.</li>"
                "<li>Stay mentally active with hobbies, puzzles, or reading.</li>"
                "</ul>"
            )
    else:
        advice_text = "No special measures required. Maintain a healthy lifestyle and schedule periodic checkups."

    return jsonify({"text": advice_text})

@app.route("/severity_page")
def severity_page():
    return render_template("predict_severity.html")

# ------------------ Run Flask App ------------------
if __name__ == "__main__":
    app.run(debug=True)
