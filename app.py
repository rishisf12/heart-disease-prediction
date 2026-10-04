import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

#Unpickling the model and scaler and columns

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "knn_heart_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
expected_columns = joblib.load(BASE_DIR / "heart_columns.pkl")

fit_y = getattr(model, "_y", None)

#The only columns the retrained scaler standardises; binaries stay 0/1.
SCALED_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

st.title("Heart Disease Check")
st.write("Fill in the patient's details, then press Predict.")

#Advanced settings, tucked away so they don't crowd the main form
with st.sidebar.expander("Advanced settings"):
    n_neighbors = st.slider(
        "Neighbours (k)",
        min_value=1,
        max_value=25,
        value=model.n_neighbors,
        step=2,
        help="How many similar past patients vote on the result.",
    )

left, right = st.columns(2)

with left:
    st.subheader("Patient")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1, key="age")
    sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
    chest_pain = st.selectbox(
        "Chest pain",
        ["Typical angina", "Non-anginal pain", "Atypical angina"],
        key="chest_pain",
    )
    exercise_angina = st.selectbox("Chest pain during exercise", ["No", "Yes"], key="exercise_angina")
    st_slope = st.selectbox("Exercise ECG trend", ["Rising", "Flat", "Falling"], key="st_slope")

with right:
    st.subheader("Readings")
    resting_bp = st.number_input(
        "Resting blood pressure", min_value=1, max_value=300, value=120, step=1, key="resting_bp"
    )
    cholesterol = st.number_input("Cholesterol", min_value=1, max_value=610, value=200, step=1, key="cholesterol")
    fasting_bs = st.selectbox("Blood sugar above 120", ["No", "Yes"], key="fasting_bs")
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "Abnormal", "Left ventricular hypertrophy"], key="resting_ecg"
    )
    max_hr = st.number_input("Max heart rate", min_value=0, max_value=250, value=150, step=1, key="max_hr")
    # Oldpeak can legitimately be negative in the source data (down to -2.6), so the
    # lower bound must allow it rather than clamping a real reading to zero.
    oldpeak = st.number_input("ST depression", min_value=-3.0, max_value=7.0, value=1.0, step=0.1, key="oldpeak")

#Map the raw inputs onto the 15 columns the model was trained on
CHEST_PAIN_MAP = {
    "Typical angina": "ATA",
    "Non-anginal pain": "NAP",
    "Atypical angina": "TA",
}
RESTING_ECG_MAP = {
    "Normal": "Normal",
    "Abnormal": "ST",
    "Left ventricular hypertrophy": "HLV",
}

chest = CHEST_PAIN_MAP[chest_pain]
ecg = RESTING_ECG_MAP[resting_ecg]

features = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": 1 if fasting_bs == "Yes" else 0,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,
    "Sex_M": 1 if sex == "Male" else 0,
    "ChestPainType_ATA": 1 if chest == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest == "NAP" else 0,
    "ChestPainType_TA": 1 if chest == "TA" else 0,
    #LV hypertrophy is the dropped category, so both flags stay 0
    "RestingECG_Normal": 1 if ecg == "Normal" else 0,
    "RestingECG_ST": 1 if ecg == "ST" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
    #Falling is the dropped category, so both flags stay 0
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Rising" else 0,
}

input_df = pd.DataFrame([{col: features[col] for col in expected_columns}])

# Guard: if a dropdown label is ever renamed without updating its mapping, the flag
# silently becomes 0 and the prediction changes with no visible error. Catch that here.
DROPDOWN_FLAGS = {
    "Sex_M": (sex, {"Male": 1, "Female": 0}),
    "ChestPainType_ATA": (chest_pain, {"Typical angina": 1, "Non-anginal pain": 0, "Atypical angina": 0}),
    "ChestPainType_NAP": (chest_pain, {"Typical angina": 0, "Non-anginal pain": 1, "Atypical angina": 0}),
    "ChestPainType_TA": (chest_pain, {"Typical angina": 0, "Non-anginal pain": 0, "Atypical angina": 1}),
    "ExerciseAngina_Y": (exercise_angina, {"No": 0, "Yes": 1}),
    "FastingBS": (fasting_bs, {"No": 0, "Yes": 1}),
    "RestingECG_Normal": (resting_ecg, {"Normal": 1, "Abnormal": 0, "Left ventricular hypertrophy": 0}),
    "RestingECG_ST": (resting_ecg, {"Normal": 0, "Abnormal": 1, "Left ventricular hypertrophy": 0}),
    "ST_Slope_Flat": (st_slope, {"Rising": 0, "Flat": 1, "Falling": 0}),
    "ST_Slope_Up": (st_slope, {"Rising": 1, "Flat": 0, "Falling": 0}),
}
for column, (chosen, allowed) in DROPDOWN_FLAGS.items():
    if chosen not in allowed:
        st.error(f"Internal error: '{chosen}' is not a valid option for {column}.")
        st.stop()

#Two artifact styles are supported:
#  - the legacy scaler was fitted on all 15 columns, so transform the whole frame
#  - the retrained scaler covers only the 5 numeric columns, so scale those and
#    leave the binary columns as 0/1
if scaler.n_features_in_ == len(expected_columns):
    scaled = scaler.transform(input_df[expected_columns])
else:
    scaled = input_df.copy()
    scaled[SCALED_COLUMNS] = scaler.transform(input_df[SCALED_COLUMNS])

#Honour the k choice without mutating the unpickled model
active_model = model if n_neighbors == model.n_neighbors else model.set_params(n_neighbors=n_neighbors)

#Run the model when Predict is pressed, or once a prediction already exists so the
#page is never blank on first load.
if st.button("Predict", type="primary", icon=":material/health_and_safety:", width="stretch"):
    st.session_state["predicted"] = True

if st.session_state.get("predicted"):
    prediction = active_model.predict(scaled)[0]
    probabilities = active_model.predict_proba(scaled)[0]
    confidence = probabilities[prediction] * 100

    #Result
    if prediction == 1:
        st.error(f"## Likely heart disease\n{confidence:.0f}% confidence")
    else:
        st.success(f"## Likely no heart disease\n{confidence:.0f}% confidence")

    st.progress(confidence / 100)

    if confidence < 60:
        st.info("The model is unsure about this one. Treat it as a rough guide only.")

    with st.expander("How this result was reached"):
        st.write(
            f"The model compared this patient with the {n_neighbors} most similar "
            "patients in its training data, then took the majority answer."
        )
        if fit_y is not None:
            distances, neighbour_idx = active_model.kneighbors(scaled)
            labels = ["Heart disease" if fit_y[i] == 1 else "No heart disease" for i in neighbour_idx[0]]
            agree = sum(1 for label in labels if (label == "Heart disease") == bool(prediction))
            st.write(f"{agree} of the {len(labels)} closest patients share this result.")
            for rank, label in enumerate(labels, start=1):
                mark = "same answer" if ((label == "Heart disease") == bool(prediction)) else "different answer"
                st.write(f"{rank}. {label} - {mark}")

if scaler.n_features_in_ == len(expected_columns):
    #Legacy artifacts: the scaler covered all 15 columns because it had been fitted
    #on already-scaled data, and it was fitted on the test split.
    st.warning(
        "This model was trained with the scaler fitted on the test split, so its "
        "accuracy is overstated. Run `python train.py --promote` to replace it with a "
        "correctly trained model."
    )
else:
    st.caption(f"Model trained on {model.n_samples_fit_} patients, k={model.n_neighbors}.")
st.caption("Educational demo only - not medical advice.")