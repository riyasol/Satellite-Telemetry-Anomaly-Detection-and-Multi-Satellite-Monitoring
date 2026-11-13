# Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring

**Course:** Introduction to Machine Learning (IML)  
**Project Category:** Application  

### 👥 Team Members & Contributions

| Member | Roll Number | Primary Contributions (per Proposal) |
| :--- | :--- | :--- |
| **Riya** | IIT2024134 | Data preprocessing, cleaning, feature engineering, dataset preparation, and telemetry input integration. |
| **Shreya** | IIT2024137 | Implementation and tuning of the Isolation Forest model, model experimentation, and performance evaluation. |
| **Gagan** | IIT2024133 | Development of rule-based anomaly detection, visualization of telemetry patterns, and baseline comparison. |
| **Jayanth** | IIT2024155 | Integration of the complete pipeline, development of the Streamlit monitoring interface, automated reporting, testing, and documentation. |

*All four members jointly contributed to experiment design, interpretation of results, documentation, and presentation.*

---

## 1. Project Overview & Motivation

Satellites continuously generate telemetry streams that capture operational health measurements, including subsystem temperatures, bus voltages, load currents, and attitude orientations. Unexpected deviations in these parameters often signify critical onboard events such as thermal control failures, battery degradation, electrical overloads, or attitude control thruster faults. 

Because manual inspection of voluminous telemetry streams is unfeasible in real-time operations, this project implements an automated, multi-satellite telemetry monitoring and anomaly detection system. The system combines:
1. **Rule-Based Threshold Detection (Baseline):** Deterministic safety boundaries for parameters with well-defined physical operating limits.
2. **Unsupervised Machine Learning (Isolation Forest):** Isolation-tree-based anomaly scoring capable of identifying multivariate correlation deviations and subtle patterns that fixed single-parameter thresholds fail to catch.
3. **Combined Monitoring Pipeline (OR-Logic):** An integrated decision system ensuring that both explicit threshold breaches and complex multivariate anomalies are detected.
4. **Dual Dataset Evaluation:** Evaluated on both multi-satellite controlled demonstration telemetry (`SAT-01`, `SAT-02`, `SAT-03`) and actual spacecraft telemetry from the **European Space Agency (ESA) Anomaly Detection Benchmark Mission 1**.
5. **Interactive Streamlit Web Dashboard:** A 5-tab mission interface displaying constellation health trends, satellite-wise telemetry, detected anomalies with violation reasons, and experimental evaluations.

---

## 2. System Architecture & Project Structure

```
Project Code/
├── data/
│   └── telemetry_data.csv                 # Multi-satellite telemetry (SAT-01, SAT-02, SAT-03; 1,500 records)
├── ESA-Mission1/ESA-Mission1/             # Real European Space Agency benchmark dataset
│   ├── channels.csv                       # Channel metadata (subsystems, physical units, groups)
│   ├── labels.csv                         # Ground-truth operational anomaly time intervals
│   ├── anomaly_types.csv                  # Anomaly taxonomy (category, dimensionality, locality)
│   └── channels/channel_*.zip             # Spacecraft telemetry channel time-series archives
├── models/
│   ├── isolation_forest_synthetic.joblib  # Trained Isolation Forest model on multi-satellite telemetry
│   └── isolation_forest_esa.joblib        # Trained Isolation Forest model on actual ESA spacecraft channels
├── reports/
│   ├── telemetry_analysis_report.csv      # Complete telemetry dataset with rule, ML, and combined predictions
│   ├── telemetry_analysis_report_synthetic.csv # Synthetic telemetry analysis report
│   ├── telemetry_analysis_report_esa.csv  # ESA Mission 1 actual telemetry analysis report
│   ├── experiment_1_overlap.csv           # Rule vs ML overlap and agreement breakdown
│   ├── experiment_1_metrics.csv           # Rule vs ML precision, recall, F1, accuracy comparison
│   ├── experiment_2_contamination.csv     # Contamination sensitivity sweep results (c = 0.01, 0.03, 0.05, 0.10)
│   ├── experiment_3_features.csv          # Feature combination ablation results (univariate vs bivariate vs multivariate)
│   ├── experiment_4_method_comparison.csv # Side-by-side performance table (Rule vs IF vs Combined)
│   └── figures/
│       ├── fig1_synthetic_telemetry.png   # Time-series telemetry plots with red anomaly markers
│       ├── fig2_confusion_matrices.png    # Side-by-side confusion matrix heatmaps
│       ├── fig3_contamination_curve.png   # Sensitivity curve across contamination levels
│       ├── fig4_feature_combinations.png  # Ablation comparison bar chart
│       ├── fig5_anomaly_scores_distribution.png # Distribution of ML anomaly scores
│       └── fig6_esa_mission1_telemetry.png# Actual ESA Mission 1 spacecraft telemetry time-series plot
├── src/
│   ├── __init__.py                        # Package marker
│   ├── generate_data.py                   # Realistic multi-satellite telemetry generation with orbital cycles
│   ├── data_loader.py                     # Dual-source data ingestion, schema validation, cleaning, and preparation
│   ├── anomaly_detector.py                # Rule-based threshold detector with violation reason tracking
│   ├── ml_anomaly_detector.py             # Scikit-learn Isolation Forest model wrapper and scoring
│   ├── evaluation.py                      # Accuracy, Precision, Recall, F1, and Confusion Matrix calculations
│   ├── experiments.py                     # Automated execution of all 4 proposal experiments
│   ├── visualizer.py                      # Matplotlib plotting functions and figure export
│   ├── main.py                            # End-to-end command-line pipeline execution script
│   └── dashboard.py                       # Interactive Streamlit monitoring dashboard (5 tabs, dual source)
├── tests/
│   ├── __init__.py                        # Package marker
│   └── test_system.py                     # Automated unit and integration test suite (8 tests)
├── .vscode/
│   └── settings.json                      # VS Code interpreter and Pylance search paths
├── .gitignore                             # Ignored files (venv, pycache, temporary artifacts)
└── README.md                              # Project documentation
```

---

## 3. Dual Dataset Support

### 1. Controlled Multi-Satellite Demonstration Telemetry
- **Constellation:** SAT-01, SAT-02, and SAT-03 sampled at 1-minute intervals (1,500 records).
- **Physics-Informed Baseline:** Nominal 90-minute orbital cycle for temperature and battery charge/discharge; 45-minute attitude oscillation.
- **Monitored Physical Parameters:**
  - `temperature` (°C): Nominal 66°C to 74°C.
  - `voltage` (V): Nominal 3.70V to 4.00V.
  - `current` (A): Nominal 1.65A to 2.05A.
  - `orientation` (°): Nominal -20° to +20°.
- **5 Controlled Injected Fault Modes:**
  1. *Thermal Spike:* Radiator / thermal control failure (`temperature > 80.0°C`).
  2. *Voltage Collapse:* Battery cell degradation (`voltage < 3.0V`).
  3. *Current Surge:* Electrical overload / short circuit (`current > 2.5A`).
  4. *Attitude Anomaly:* Reaction wheel desaturation / tumble (`|orientation| > 45.0°`).
  5. *Multivariate Drift:* Sensor correlation violation where each individual telemetry parameter stays within its permissible threshold, but their joint behavior violates physical laws.

### 2. ESA Anomaly Detection Benchmark Mission 1 (Actual Spacecraft Telemetry)
- **Real Spacecraft Telemetry:** Evaluates real flight telemetry from European Space Agency Mission 1 (Subsystem 6 Group 3: `channel_12`, `channel_13`, `channel_19`, `channel_20`).
- **Data Ingestion:** Handles compressed channel archives, pickled time-series alignment, and missing value interpolation.
- **Ground-Truth Label Mapping:** Maps true anomaly start/end intervals from `labels.csv` to telemetry rows.
- **Statistical Control Thresholds:** Dynamically derives empirical 3-sigma boundaries from nominal baseline operations for uncalibrated spacecraft channels.

---

## 4. Methodology & Implementation

### 1. Data Cleaning & Feature Preparation (Riya)
- Ingests raw telemetry records from either synthetic multi-satellite CSV or ESA channel archives.
- Standardizes timestamps, enforces chronological ordering, and imputes missing readings using time-based interpolation.
- Prepares numerical feature matrices for training and inference.

### 2. Rule-Based Threshold Detection (Gagan)
- Evaluates individual telemetry parameters against configured upper and lower safety boundaries.
- For synthetic data, applies engineering limits (`temp > 80°C`, `voltage < 3.0V`, `current > 2.5A`, `|orientation| > 45°`).
- For ESA telemetry, computes dynamic 3-sigma statistical thresholds from baseline nominal periods.
- Records exact parameter violation descriptions (e.g. `temperature=88.50 > max(80.00)`).

### 3. Isolation Forest Machine Learning (Shreya)
- Implements genuine unsupervised anomaly detection using `sklearn.ensemble.IsolationForest`.
- Uses an ensemble of isolation trees ($i\text{Trees}$) that partition feature space; anomalies are isolated near tree roots.
- Trained strictly on nominal baseline observations to prevent temporal data leakage.
- Converts raw decision function outputs to continuous, positive **ML Anomaly Scores** (higher values = higher likelihood of anomaly).
- Persists trained models to disk using `joblib`.

### 4. Combined Detection Strategy (Jayanth)
- Fuses detections through logical OR:
  $$\text{Combined Anomaly} = (\text{Rule Flag} == 1) \lor (\text{ML Flag} == 1)$$
- Classifies each observation into: `"Both (Rule + ML)"`, `"Rule Anomaly"`, `"ML Anomaly"`, or `"Normal"`.
- Combines the high precision of safety thresholds with the high recall of multi-sensor ML.

---

## 5. Proposal Experiments & Results

All four experiments described in Section 3 of the project proposal are programmatically executed on chronologically held-out unseen test partitions:

### Experiment 1: Rule-Based vs. Isolation Forest Overlap & Agreement
Evaluates the agreement and complementary strengths between threshold rules and Isolation Forest:
- **Agreed Normal Observations:** 428 samples (95.11%)
- **Agreed Anomalies (Both):** 13 samples (2.89%)
- **Rule-Only Detections (Single-Sensor Outliers):** 3 samples (0.67%)
- **ML-Only Detections (Multivariate Drifts):** 6 samples (1.33%)
- **Overall Decision Agreement:** **98.00%**

### Experiment 2: Contamination Parameter Sensitivity Sweep
Tests contamination values $c \in \{0.01, 0.03, 0.05, 0.10\}$ on unseen test observations:

| Contamination ($c$) | Anomalies Flagged | Flagged Rate (%) | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | 0 | 0.00% | 0.9578 | 0.0000 | 0.0000 | 0.0000 |
| **0.03** | 8 | 1.78% | 0.9756 | 1.0000 | 0.4211 | 0.5926 |
| **0.05 (Nominal)** | 19 | 4.22% | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **0.10** | 40 | 8.89% | 0.9533 | 0.4750 | 1.0000 | 0.6441 |

### Experiment 3: Feature Combination Ablation Study
Studies performance across single parameters, sensor pairs, and the full feature vector:

| Feature Combination | Parameters Used | No. of Features | Flagged | Precision | Recall | F1-Score | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Univariate: Temperature** | `temperature` | 1 | 18 | 0.4444 | 0.4211 | 0.4324 | 0.9533 |
| **Univariate: Voltage** | `voltage` | 1 | 17 | 0.4706 | 0.4211 | 0.4444 | 0.9556 |
| **Univariate: Current** | `current` | 1 | 19 | 0.3684 | 0.3684 | 0.3684 | 0.9467 |
| **Bivariate Pair** | `temperature`, `voltage` | 2 | 20 | 0.6500 | 0.6842 | 0.6667 | 0.9711 |
| **Full Multivariate Set** | `temp`, `volt`, `curr`, `orient` | 4 | 19 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Experiment 4: Comprehensive Method Performance Evaluation
Side-by-side performance comparison on the unseen test partition:

| Method | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN | Total Flagged |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | 0.9800 | 0.8125 | 0.6842 | 0.7429 | 13 | 3 | 428 | 6 | 16 |
| **Isolation Forest (ML)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 19 | 0 | 431 | 0 | 19 |
| **Combined (OR-Logic)** | 0.9933 | 0.8636 | **1.0000** | 0.9268 | 19 | 3 | 428 | 0 | 22 |

---

## 6. Interactive Streamlit Dashboard

The Streamlit web interface (`src/dashboard.py`) provides operators with 5 dedicated sections and a sidebar toggle between **Synthetic Telemetry** and **ESA Mission 1 Benchmark**:
1. **Overview:** Constellation-wide health counters, total observations, normal readings, detected anomalies, anomaly percentage, and detector breakdown.
2. **Dataset:** Tabular telemetry preview, data types, min/max values, and descriptive statistics.
3. **Anomaly Detection:** Rule-based, Isolation Forest, and Combined detection tables with violation reasons and satellite filters.
4. **Experiments & Evaluation:** Interactive tabs displaying results for Experiments 1 through 4, including confusion matrix heatmaps and sensitivity curves.
5. **Visualizations & Reports:** Telemetry time-series charts with red markers highlighting anomalies, continuous ML anomaly score histograms, and one-click CSV report downloads.

---

## 7. How to Run the Project

### Prerequisites
- Python 3.10+ (or the pre-configured project virtual environment)
- Standard packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `joblib`

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

### 1. Run the End-to-End CLI Pipeline
Executes both synthetic multi-satellite and actual ESA telemetry pipelines, trains models, evaluates test sets, runs all 4 experiments, and exports CSV reports and figures:
```powershell
.\venv\Scripts\python src\main.py
```

### 2. Launch the Streamlit Monitoring Dashboard
Launches the interactive web dashboard with dual dataset support in your default browser:
```powershell
.\venv\Scripts\streamlit run src\dashboard.py
```

### 3. Run the Automated Test Suite
Verifies all 8 unit and integration tests across data processing, ESA loading, detectors, evaluation, experiments, and plotting:
```powershell
.\venv\Scripts\python -m unittest tests\test_system.py
```
