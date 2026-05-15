# 🏎️ F1 Podium Predictor — ANN-Based Race Outcome Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **Can an Artificial Neural Network predict who stands on the Formula 1 podium?**  
> This project answers that question by building, training, and comparing three distinct ANN architectures on historical F1 race data — all wrapped in an interactive Streamlit dashboard with a race-night aesthetic.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Architecture](#-architecture)
4. [Dataset](#-dataset)
5. [Models](#-models)
6. [Performance](#-performance)
7. [Project Structure](#-project-structure)
8. [Installation & Setup](#-installation--setup)
9. [Usage](#-usage)
10. [Tech Stack](#-tech-stack)

---

## 🔍 Project Overview

Formula 1 podium prediction is a classic binary classification challenge: given pre-race information about a driver, their constructor, and qualifying results, can we predict whether they will finish in the top 3?

This project tackles that challenge end-to-end:

- **Data pipeline** — merges race results, starting grids, and qualifying sheets from multiple historical CSV files
- **Feature engineering** — label encoding of categorical fields (driver, constructor, circuit) + StandardScaler normalization
- **Model training** — three ANN architectures trained with binary cross-entropy loss and evaluated on five metrics
- **Interactive dashboard** — Streamlit app with real-time prediction, animated gauge charts, and side-by-side model comparison

The result is a fully reproducible ML project that achieves **~89% accuracy** and **AUC ≈ 0.89** across all three architectures.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔄 **End-to-end pipeline** | Raw CSVs → preprocessed features → trained models in two script runs |
| 🧠 **3 ANN architectures** | Baseline, complex, and dropout-regularized networks compared head-to-head |
| 📊 **5-metric evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC reported per model |
| 📉 **Loss curve visualisation** | Training convergence plotted for every model |
| 🎛️ **Interactive dashboard** | Select any model, any driver, any Grand Prix, and get an instant probability score |
| 🏁 **F1-themed UI** | Custom CSS with F1 red (`#e10600`), dark backgrounds, and Plotly gauge charts |

---

## 🏗️ Architecture

```
F1_podium_predictor/
├── data/                    ← Raw + processed CSV datasets
├── models/                  ← Trained .keras models + sklearn artifacts (.pkl)
├── src/
│   ├── data_preprocessing.py   ← Feature engineering & normalization pipeline
│   └── model_training.py       ← ANN definitions, training loop, evaluation
├── app/
│   └── main.py              ← Streamlit dashboard
├── assets/                  ← Static images for the UI
├── requirements.txt
└── README.md
```

### Data Flow

```
Raw CSVs  ──►  data_preprocessing.py  ──►  processed_f1_data.csv
                                               │
                                               ▼
                             model_training.py (trains A, B, C)
                                               │
                                     ┌─────────┴─────────┐
                                     ▼                   ▼
                              models/*.keras       models/*.pkl
                                               │
                                               ▼
                                   app/main.py (Streamlit)
```

---

## 📦 Dataset

The dataset spans **multiple decades of Formula 1 history** and is composed of the following files inside `data/`:

| File | Contents |
|---|---|
| `race_details.csv` | Race results — finishing position per driver per event |
| `starting_grids.csv` | Grid positions at race start |
| `qualifyings.csv` | Official qualifying session positions |
| `driver_standings.csv` | Championship standings snapshot per round |
| `constructor_standings.csv` | Constructor championship standings |
| `fastest_laps.csv` | Fastest lap records per race |
| `pitstops.csv` | Pit-stop timing data |
| `practices.csv` | Practice session lap times |
| `race_summaries.csv` | High-level race metadata |
| `driver_details.csv` | Driver biographical information |
| `team_details.csv` | Constructor/team metadata |

### Preprocessing Steps (`src/data_preprocessing.py`)

1. **Load** `race_details`, `starting_grids`, and `qualifyings` CSVs.
2. **Create binary target**: `Podium = 1` if `finishing_position ≤ 3`, else `0`.
3. **Left-merge** the three tables on `[Year, Grand Prix, Driver]`; missing values filled with `20` (last place proxy).
4. **Label-encode** categorical columns: `Driver`, `Grand Prix`, `Car` (constructor).  
   Encoders are persisted to `models/le_*.pkl` for reuse at inference time.
5. **Feature set** (6 features):  
   `Year`, `gp_encoded`, `driver_encoded`, `car_encoded`, `grid_position`, `qualifying_position`
6. **Normalise** with `StandardScaler` (mean = 0, std = 1). Scaler saved to `models/scaler.pkl`.
7. Write `data/processed_f1_data.csv`.

---

## 🧠 Models

Three feed-forward ANN architectures are defined in `src/model_training.py` and trained with:

- **Optimiser**: Adam  
- **Loss**: Binary Cross-Entropy  
- **Epochs**: 50 | **Batch size**: 32 | **Validation split**: 10%

### Model A — Baseline
```
Input(6) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```
Lightweight two-hidden-layer network, establishes the performance baseline.

### Model B — Complex
```
Input(6) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)
```
Wider network with more parameters; tests whether additional capacity improves prediction.

### Model C — Regularized (Dropout)
```
Input(6) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(1, Sigmoid)
```
Adds a 30% dropout layer after the first hidden layer to reduce overfitting.

---

## 📈 Performance

All three models were evaluated on a held-out **20% test split** (stratified, `random_state=42`):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Model A** (Baseline) | 89.01% | 60.22% | 43.63% | 50.60% | 0.8923 |
| **Model B** (Complex) | 89.19% | 63.71% | 37.74% | 47.40% | 0.8850 |
| **Model C** (Regularized) | 89.26% | 65.58% | 35.19% | 45.80% | 0.8914 |

**Key observations:**

- All models achieve near-identical **accuracy (~89%)** and **AUC (~0.89)**, indicating strong and consistent discrimination ability.
- **Model A** achieves the highest **F1 score (50.6%)**, making it the best balanced classifier.
- **Model B & C** trade recall for precision — they are more conservative but more accurate when they do predict a podium.
- The class imbalance inherent in F1 (only 3 out of ~20 drivers podium per race) suppresses recall across all models.

---

## 🗂️ Project Structure

```
F1_podium_predictor/
│
├── app/
│   └── main.py                  # Streamlit dashboard app
│
├── assets/
│   └── images/
│       └── background.png       # F1 background image for UI
│
├── data/
│   ├── race_details.csv
│   ├── starting_grids.csv
│   ├── qualifyings.csv
│   ├── driver_standings.csv
│   ├── constructor_standings.csv
│   ├── fastest_laps.csv
│   ├── fastestlaps_detailed.csv
│   ├── pitstops.csv
│   ├── practices.csv
│   ├── race_summaries.csv
│   ├── driver_details.csv
│   ├── team_details.csv
│   ├── sprint_grid.csv
│   ├── sprint_results.csv
│   └── processed_f1_data.csv    # Generated by preprocessing script
│
├── models/
│   ├── Model_A.keras
│   ├── Model_B.keras
│   ├── Model_C.keras
│   ├── le_driver.pkl
│   ├── le_grand_prix.pkl
│   ├── le_car.pkl
│   ├── scaler.pkl
│   └── training_results.pkl
│
├── src/
│   ├── data_preprocessing.py    # Feature engineering pipeline
│   └── model_training.py        # ANN definitions & training loop
│
├── models performance.csv        # Summary of test-set metrics
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.10+**
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/shahdelhadad/F1-ANN-model.git
cd F1-ANN-model
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Preprocess the data

> Only required if `data/processed_f1_data.csv` does not exist.

```bash
python src/data_preprocessing.py
```

This reads the raw CSVs from `data/`, merges them, encodes features, fits the scaler, and writes `data/processed_f1_data.csv` + all `.pkl` artifacts to `models/`.

### Step 2 — Train the models

> Only required if the `.keras` files do not exist in `models/`.

```bash
python src/model_training.py
```

This trains all three ANN architectures for 50 epochs each, prints a comparison table, and saves the models to `models/`.

### Step 3 — Launch the dashboard

```bash
streamlit run app/main.py
```

The app will open in your browser at **`http://localhost:8501`**.

#### Dashboard walkthrough

| Panel | What it does |
|---|---|
| **Sidebar** | Select ANN architecture (A / B / C) |
| **Race Parameters** | Choose year, Grand Prix, driver, and constructor |
| **Qualifying Data** | Enter grid and qualifying positions |
| **Predict button** | Runs inference and displays a Plotly gauge chart with probability % |
| **Model Performance** | Live metrics table + grouped bar chart + training loss curves |

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `TensorFlow / Keras` | ANN model definition, training, serialisation |
| `scikit-learn` | Label encoding, StandardScaler, train/test split, metrics |
| `pandas` | Data loading, merging, feature construction |
| `numpy` | Numerical operations |
| `Streamlit` | Interactive web dashboard |
| `Plotly` | Gauge charts, bar charts, loss curve visualisation |
| `joblib` | Serialising sklearn transformers and result dicts |
| `matplotlib` | Auxiliary plotting |
| `Pillow` | Image handling in Streamlit |

---

## 📄 License

This project is released under the [MIT License](LICENSE).


