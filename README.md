# 🧠 NAFLD_XAI — Explainable AI for NAFLD Risk Prediction (Clinical-Ready API)

> **A production-oriented ML + FastAPI system that not only predicts NAFLD risk (~80%+ accuracy) but explains *why*—feature by feature—using SHAP. Built for real-world healthcare trust, not just accuracy.**

---

## 🔥 Why This Project Matters (Strong Overview)

Most ML models in healthcare fail at one critical point: **trust**.
Doctors don’t just need predictions — they need **reasoning**.

**NAFLD_XAI solves this gap by combining:**

* 🧠 **High-performance ML model** (80%+ accuracy)
* 🔍 **Explainable AI (XAI)** using SHAP
* ⚡ **Production-ready FastAPI backend**
* 🏥 **Clinically meaningful features and outputs**

➡️ This transforms the model from a *black box* into a **decision-support system**.

---

## 🚀 Key Innovations / Uniqueness

### 1️⃣ Explainability First (Not an Afterthought)

* Uses **SHAP TreeExplainer** directly in API pipeline
* Returns **feature-level contribution values** per patient
* Enables:

  * Model transparency
  * Clinical validation
  * Bias detection

---

### 2️⃣ Real-Time ML + XAI API

* Built with **FastAPI inside `main.py`**
* Supports:

  * Real-time predictions
  * Instant explainability
* Swagger UI auto-docs included

---

### 3️⃣ Robust Feature Engineering Pipeline

* Uses saved artifacts:

  * `model.pkl`
  * `scaler.pkl`
  * `features.pkl`
* Ensures:

  * Consistent inference
  * No training-serving skew

---

### 4️⃣ Clinically-Aligned Risk Output

Instead of raw numbers, API returns:

* ✔️ Risk classification (**High / Low Risk**)
* ✔️ Probability score
* ✔️ Feature contribution breakdown
* ✔️ Medical disclaimer

---

### 5️⃣ Fault-Tolerant XAI Handling

Handles multiple SHAP output formats safely:

* Binary classifier outputs
* Flattened feature contributions
* Prevents runtime crashes in production

---

## 📂 Project Structure

```
NAFLD_XAI/
│
├── NAFLD.xlsx        # Dataset
├── model.pkl         # Trained model
├── scaler.pkl        # Preprocessing scaler
├── features.pkl      # Feature list
├── main.py           # FastAPI + prediction + SHAP
├── train_model.py    # Model training pipeline
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/NishiKumariPatel/NAFLD_XAI.git
cd NAFLD_XAI
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn main:app --reload
```

📍 Open in browser:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  (Swagger UI)
* [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📥 API Request Example

```json
POST /predict
{
  "age": 45,
  "gender": 1,
  "bmi": 29.5,
  "waist": 102,
  "ast": 40,
  "alt": 55,
  "glucose": 110,
  "hba1c": 6.2,
  "triglycerides": 180,
  "hdl": 38,
  "ldl": 130,
  "insulin": 15,
  "homa": 3.2
}
```

---

## 📤 API Response Example

```json
{
  "fibrosis_prediction": 1,
  "probability": 0.82,
  "risk_level": "High Risk",
  "feature_contributions": {
    "bmi": 0.34,
    "waist": 0.21,
    "triglycerides": 0.18
  },
  "note": "This is a screening tool. Consult a medical professional."
}
```

---

## 🧠 How It Works (Pipeline)

### Step 1: Input Handling

* Receives structured patient data via API
* Converts into DataFrame aligned with training features

### Step 2: Preprocessing

* Applies `scaler.pkl` to normalize input

### Step 3: Prediction

* Uses trained ML model (`model.pkl`)
* Outputs:

  * Binary classification
  * Probability score

### Step 4: Explainability (XAI)

* SHAP computes contribution of each feature
* Outputs interpretable values per prediction

---

## 📊 Model Performance

* Accuracy: **80%+**
* Balanced clinical feature usage
* Generalizable pipeline design

---

## 🔍 Explainability (Deep Insight)

Unlike typical ML APIs, this system returns:

* 📈 Positive SHAP values → increase NAFLD risk
* 📉 Negative SHAP values → decrease risk

➡️ This enables:

* Feature impact tracing
* Personalized patient analysis
* Trustworthy AI decisions

---

## ⚠️ Medical Disclaimer

This project is intended for **research and screening purposes only**.
It is **not a diagnostic tool**.

Always consult a qualified medical professional.

---

## 💡 Future Scope

* 🌐 Deploy on cloud (AWS / Render)
* 📊 Add SHAP visualization dashboard
* 🧬 Integrate clinical datasets
* 🤖 Improve model to 85%+ accuracy
* 📱 Build frontend (React / Streamlit)

---

## 🤝 Contributing

Pull requests and improvements are welcome.

---

## 📬 Contact

GitHub: [https://github.com/NishiKumariPatel](https://github.com/NishiKumariPatel)

---

⭐ **If this project helped you, consider giving it a star!**
