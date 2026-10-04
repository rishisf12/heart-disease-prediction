# Heart Disease Prediction (KNN)

A Streamlit web app that predicts the presence of heart disease using a
**K-Nearest Neighbours** classifier trained on the Heart Disease dataset
(918 patients, 11 raw features).

## What it does

Enter a patient's details, press **Predict**, and the app returns a
diagnosis with a confidence score, plus an explanation of which training
patients drove the result.

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application |
| `knn_heart_model.pkl` | Fitted `KNeighborsClassifier` |
| `scaler.pkl` | Fitted `StandardScaler` |
| `heart_columns.pkl` | Expected feature column order |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## How the model works

KNN makes no arithmetic prediction of its own. For a new patient it measures
the distance to every training patient in standardised space, takes the `k`
closest, and returns their majority diagnosis. The app shows this directly —
"3 of the 5 closest patients share this result" — so a prediction is always
traceable back to real patients rather than an opaque score.

### Feature encoding

The 11 raw features expand to 15 model inputs via one-hot encoding:

- `Sex` → `Sex_M`
- `ChestPainType` → `ChestPainType_ATA`, `_NAP`, `_TA`
- `RestingECG` → `RestingECG_Normal`, `_ST`
- `ExerciseAngina` → `ExerciseAngina_Y`
- `ST_Slope` → `ST_Slope_Flat`, `_Up`

Encoding was created with `drop_first=True`, so two categories are absent and
must be represented by an all-zero row:

- **LV hypertrophy** — `RestingECG_Normal=0` *and* `RestingECG_ST=0`
- **Falling ST slope** — `ST_Slope_Flat=0` *and* `ST_Slope_Up=0`

The app encodes these deliberately. It also validates every dropdown against
its column mapping at runtime and blocks the prediction on mismatch, because a
mismatched flag would otherwise change the result silently.

### Input ranges

The numeric bounds in the UI match the source dataset. Note that `Oldpeak` may
be **negative** — a real finding meaning ST elevation rather than depression —
and the input allows it:

| Field | Dataset range | Accepted in app |
|---|---|---|
| `Age` | 28 – 77 | 1 – 120 |
| `RestingBP` | 0 – 200 | 1 – 300 |
| `Cholesterol` | 0 – 603 | 1 – 610 |
| `MaxHR` | 60 – 202 | 0 – 250 |
| `Oldpeak` | **-2.6** – 6.2 | -3.0 – 7.0 |

`RestingBP` and `Cholesterol` start at 1 rather than 0 because training replaced
zero readings with the column mean, so the model has never seen a literal `0`
for either field. Entering one would land far off-manifold. If a reading is
genuinely unknown, leave the default.

## Known issues

**The bundled model has a data-leakage bug. Do not trust its accuracy.**

Reading directly from the artifacts:

- `scaler.n_samples_seen_` = **184**
- `model.n_samples_fit_` = **734**
- 184 + 734 = 918 = the full dataset

The scaler was fitted on the **test** split while the model was fitted on the
**train** split. Any accuracy figure derived from this pipeline is optimistic
and meaningless.

A second issue compounds it: `scaler.scale_[Age]` is `1.109`, but raw `Age` has
a standard deviation of `9.43`. The scaler was never fitted on raw values,
which means applying `scaler.transform()` to a raw patient produces wildly
inflated values (a raw `Age` of 50 maps to z ≈ 45, and the nearest training
patient sits at a distance of ~237).

Corrected pipeline:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)   # fit on TRAIN only
model.fit(scaler.transform(X_train), y_train)   # model sees TRAIN
```

Additional preprocessing concerns worth fixing at the same time:

- Binary features were standardised, including every dummy. Since KNN relies on
  Euclidean distance, feature scale directly changes which patients are grouped.
  Standardising only the five numeric columns and leaving dummies as 0/1 is the
  conventional fix.
- `Cholesterol == 0` (172 rows, 18.7%) was treated as missing and replaced with
  the column mean of `244.64`, conflating "not measured" with "actually 244".
  A median or an explicit missing-value indicator is more honest.

Until this is resolved, treat the app as a UI demonstration rather than a
working classifier.

## Data

Source: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease),
as commonly redistributed on Kaggle. Features are anonymised patient records;
this is for education and demonstration, **not for medical use**.