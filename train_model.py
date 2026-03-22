import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel("NAFLD.xlsx")

# Robust column normalization: lowercase, replace spaces/dashes with single underscore
df.columns = df.columns.str.strip().str.lower().str.replace(r"[ -]+", "_", regex=True)

print("Columns after normalization:", df.columns.tolist())
print("Dataset loaded ✅")

# ==============================
# FEATURES AND TARGET
# ==============================
feature_columns = [
    "age",
    "body_mass_index",
    "waist_circumference",
    "ast",
    "alt",
    "triglycerides",
    "hdl",
    "ldl",
    "total_cholesterol",
    "glucose",
    "hemoglobin_a1c",
    "systolic_blood_pressure",
    "diastolic_blood_pressure"
]

# Drop rows with missing values
df = df.dropna(subset=feature_columns + ["fibrosis"])

# Convert fibrosis to binary
df["fibrosis_binary"] = df["fibrosis"].apply(lambda x: 0 if x <= 1 else 1)

X = df[feature_columns]
y = df["fibrosis_binary"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# HANDLE IMBALANCE WITH SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# ==============================
# TRAIN RANDOM FOREST MODEL
# ==============================
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train_res, y_train_res)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL, SCALER, FEATURES
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "features.pkl")

print("\nModel, scaler, and features saved ✅")