# Adaptive Fusion of EEG and NIRS with Explainable AI Reveals Neurophysiological Markers of Cognitive Flexibility

## 🧠 Overview

This repository hosts the complete codebase and datasets for our research study: **"Adaptive Fusion of EEG and NIRS with Explainable AI Reveals Neurophysiological Markers of Cognitive Flexibility"**

Our work investigates how **electroencephalography (EEG)** and **near-infrared spectroscopy (fNIRS)**, combined through advanced machine learning and explainable AI, can reveal robust biomarkers of cognitive flexibility during probabilistic reversal learning tasks.

### 🎯 Key Findings
- **95.00% accuracy** achieved with dynamic gating (±6.1% variance) — **+14% improvement** over best unimodal
- **EEG variance** emerges as most discriminative feature (SHAP: 0.185)
- **42% increase** in oxygen consumption during exploratory cognitive states (Cohen's d=0.39)
- **53% variance reduction** compared to unimodal EEG baseline
- **Large effect sizes** for EEG metrics (Cohen's d: 0.79-0.88) vs. medium for fNIRS (d: 0.38-0.39)

---

## 🎯 Research Objectives

We aimed to develop and validate multimodal EEG-fNIRS biomarkers of cognitive flexibility by:

1. **Evaluating** EEG vs fNIRS discriminative capacity for goal-directed (AO) vs exploratory (SO) cognitive states
2. **Comparing** classical ML models (KNN, SVM, RF, XGBoost, LR, LightGBM) with deep neural networks (MLP, GRU, BiLSTM)
3. **Investigating** fusion strategies: early (signal-level), intermediate (feature-level), and late (decision-level) fusion
4. **Applying** SHAP for interpretability and neurophysiological marker identification

### Research Questions Addressed
- **RQ1**: Can EEG and fNIRS reliably discriminate AO vs SO cognitive states?
  - ✅ **Answer**: Yes—EEG achieves 81.0% accuracy, fNIRS 64.7%, with statistically significant differences (p<0.001)
- **RQ2**: Do multimodal fusion strategies improve classification over unimodal approaches?
  - ✅ **Answer**: Yes—dynamic gating achieves 95.00% (+14% over best unimodal, +19.7% over static fusion)
- **RQ3**: What are the strengths/weaknesses of different fusion strategies?
  - ✅ **Answer**: Early/intermediate fusion underperformed (74.2%, 75.3%); late fusion with dynamic gating optimal
- **RQ4**: How does SHAP enhance interpretability of biomarkers?
  - ✅ **Answer**: SHAP revealed EEG variance as top feature (0.185), linking to active inference theory

---

## 🔄 Methodology & Results

### Experimental Design
- **Dual-task paradigm**: Probabilistic Reversal Learning + Auditory Oddball
- **Participants**: 42 healthy young adults (mean age: 19.1 ± 2.1 years; 22 females, 20 males)
  - AO group (n=21): With specific performance objectives
  - SO group (n=21): Without specific objectives ("do your best")
- **Data modalities**: 
  - EEG: 8 electrodes, 512 Hz sampling, 24 features (time + frequency domains)
  - fNIRS: 32 detectors, 4 Hz sampling, 15 hemodynamic features
- **Validation**: Stratified 5-fold cross-validation with Wilson 95% confidence intervals

### Performance Hierarchy

| Approach | Best Model | Accuracy | F1-Score | 95% CI (Wilson) | CV Variance |
|----------|-----------|----------|----------|-----------------|-------------|
| **Unimodal EEG** | Logistic Regression | 81.0% ± 9.8% | 83.3% ± 8.9% | [0.679, 0.895] | 12.1% |
| **Unimodal fNIRS** | Random Forest | 64.7% ± 17.4% | 65.3% ± 22.8% | [0.492, 0.778] | 26.9% |
| **Early Fusion** | SVM | 74.2% ± 16.0% | 74.2% ± 12.6% | [0.572, 0.839] | 21.6% |
| **Intermediate Fusion** | XGBoost | 75.3% ± 11.5% | 76.7% ± 13.0% | [0.607, 0.862] | 15.3% |
| **Late Fusion (Baseline)** | Weighted Average | 78.1% ± 20.0% | 73.4% ± 25.8% | [0.598, 0.869] | 25.6% |
| **Late Fusion (Advanced)** | Dynamic Confidence | 80.6% ± 18.5% | 78.5% ± 22.0% | [0.650, 0.901] | 22.9% |
| **Late Fusion (Optimal)** | **Dynamic Gating** | **95.0% ± 6.1%** | **57.2% ± 12.1%** | **[0.839, 0.987]** | **6.4%** |

**Key Performance Insights:**
- **+14.0% absolute improvement** over best unimodal (EEG: 81.0% → 95.0%)
- **+19.7% absolute improvement** over best static fusion (Intermediate: 75.3% → 95.0%)
- **53% variance reduction** compared to EEG baseline (12.1% → 6.4%)
- **Note**: Accuracy-F1 discrepancy (95.0% vs 57.2%) requires investigation—see Section 4.6.2 of manuscript

### Neurophysiological Insights
- **EEG findings**: 48% increase in signal variance for exploratory states
- **NIRS findings**: 42% increase in oxygen consumption during exploration
- **Convergent evidence**: Electrical hyperactivation correlates with metabolic demands

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── Congnitive_Load_Analysis/
│   └── Cognitive_load.ipynb                
├── NIRS_Group_Type_Prediction/
│   ├── NIRS_Dataset/
│   │   ├── Dataset_overview.png
│   │   └── extrait.csv
│   ├── figures/
│   │   ├── model_performance.png
│   ├── notebook/
│   │   └── Prediction_of_Group_NIRS.ipynb
│   └── README.md
├── Fusion_EEG&NIRS_Group_Type_Prediction/
│   ├── Early_Fusion/
│   │   ├── notebooks/
│   │   │    ├── Early_fusion.ipynb
│   │   │    └── EEG_NIRS_Construct.ipynb
│   │   └── Fusion_Data
│   │         ├── Dataset_overview
│   │         └── extrait.csv       
│   ├── Features_Fusion/
│   │   └── Features_fusion.ipynb     
│   ├── Late_Fusion/
│   │   └── Late_fusion.ipynb
│   └── README.md

```

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/eeg-nirs-cognitive-flexibility.git
cd eeg-nirs-cognitive-flexibility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## 📊 Key Results

### Feature Importance (SHAP Analysis)

#### EEG Features (Modality Importance: 0.045)
| Rank | Feature | SHAP Value | Interpretation |
|------|---------|------------|----------------|
| 1 | **Variance** | **0.185** | Neural instability during exploration—indexes hypothesis space breadth |
| 2 | Alpha Power | 0.142 | Attentional suppression and inhibitory control |
| 3 | Theta Power | 0.135 | Working memory demands for maintaining multiple hypotheses |
| 4 | RMS Amplitude | 0.128 | Overall neural activation intensity |
| 5 | Delta Power | 0.128 | Motivational salience and reward processing |
| 6 | Spectral Entropy | 0.095 | Temporal complexity of neural dynamics |

#### fNIRS Features (Modality Importance: 0.028)
| Rank | Feature | SHAP Value | Interpretation |
|------|---------|------------|----------------|
| 1 | **SNR** | **0.089** | Signal quality—primary determinant for fNIRS classification |
| 2 | Spectral Entropy | 0.067 | Complexity of hemodynamic oscillations |
| 3 | Crossing Points | 0.058 | Temporal dynamics of HbO/HbR interactions |
| 4 | HbR Mean | 0.045 | Deoxygenated hemoglobin baseline levels |

**Key Insight**: EEG variance (0.185) contributes **6.6× more** to classification than top fNIRS feature (SNR: 0.089), explaining modality asymmetry in gating weights (EEG: 0.650 vs. fNIRS: 0.350).

### Neurophysiological Findings
- **Cognitive Load**: 48% increase in EEG variance during exploratory states
- **Metabolic Demand**: 42% increase in oxygen consumption
- **Neural Efficiency**: Goal-directed processing requires less neural resources

---

## 🔬 Methodology Details

### Data Acquisition
- **EEG**: 8-electrode system, 512 Hz sampling rate
- **NIRS**: 16 LEDs × 32 detectors × 2 wavelengths, 4 Hz sampling
- **Task**: Dual-task paradigm (probabilistic reversal learning + auditory oddball)

### Machine Learning Pipeline
1. **Preprocessing**: Signal filtering, artifact removal, temporal alignment
2. **Feature Engineering**: Time-domain, frequency-domain, hemodynamic features
3. **Model Training**: 8 algorithms × 3 fusion strategies × 5-fold CV
4. **Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC
5. **Interpretability**: SHAP analysis for feature importance and model explainability

### Innovation: Dynamic Gating
Our novel **dynamic gating** approach learns instance-specific fusion weights using:
- 11-dimensional meta-feature space
- Gradient Boosting Regressor for weight prediction
- Context-adaptive fusion based on prediction confidence and disagreement

---

## 📈 Clinical & Practical Applications

### Brain-Computer Interfaces
- **Adaptive systems** that adjust task difficulty based on real-time cognitive load
- **Personalized interfaces** using individual neurophysiological patterns

### Clinical Assessment
- **Cognitive flexibility screening** using EEG variance as rapid biomarker
- **Rehabilitation monitoring** for executive function recovery
- **Early detection** of cognitive impairments affecting goal-setting abilities

### Research Applications
- **Multimodal fusion** framework for other neuroimaging combinations
- **Explainable AI** approaches for transparent neurophysiological analysis
- **Active inference** validation in cognitive neuroscience

---
