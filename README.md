# Adaptive Fusion of EEG and NIRS with Explainable AI Reveals Neurophysiological Markers of Cognitive Flexibility

## 🧠 Overview

This repository hosts the complete codebase and datasets for our research study: **"Adaptive Fusion of EEG and NIRS with Explainable AI Reveals Neurophysiological Markers of Cognitive Flexibility"**

Our work investigates how **electroencephalography (EEG)** and **near-infrared spectroscopy (NIRS)**, combined through advanced machine learning and explainable AI, can reveal robust biomarkers of cognitive flexibility during probabilistic reversal learning tasks.

### 🎯 Key Findings
- **87.78% accuracy** achieved with late fusion + dynamic gating
- **EEG variance** emerges as most discriminative feature (59.2% importance)
- **42% increase** in oxygen consumption during exploratory cognitive states
- **55.4% variance reduction** compared to baseline fusion methods

---

## 🎯 Research Objectives

We aimed to develop and validate multimodal EEG-NIRS biomarkers of cognitive flexibility by:

1. **Evaluating** EEG vs NIRS discriminative capacity for goal-directed (AO) vs exploratory (SO) cognitive states
2. **Comparing** classical ML models with deep neural networks
3. **Investigating** fusion strategies: early, intermediate, and late fusion
4. **Applying** SHAP for interpretability and neurophysiological marker identification

### Research Questions Addressed
- **RQ1**: Can EEG and NIRS reliably discriminate AO vs SO cognitive states?
- **RQ2**: Do multimodal fusion strategies improve classification over unimodal approaches?
- **RQ3**: What are the strengths/weaknesses of different fusion strategies?
- **RQ4**: How does SHAP enhance interpretability of biomarkers?

---

## 🔄 Methodology & Results

### Experimental Design
- **Dual-task paradigm**: Probabilistic Reversal Learning + Auditory Oddball
- **Participants**: 42 participants (21 AO, 21 SO groups)
- **Data modalities**: EEG (8 electrodes, 512 Hz) + NIRS (1024 channels, 4 Hz)

### Performance Hierarchy
| Approach | EEG Accuracy | NIRS Accuracy | Multimodal Accuracy |
|----------|--------------|---------------|-------------------|
| **Unimodal Best** | 81.0% (Logistic Regression) | 64.7% (Random Forest) | - |
| **Early Fusion** | - | - | 66.1% (SVM) |
| **Intermediate Fusion** | - | - | 70.6% (Logistic Regression) |
| **Late Fusion + Dynamic Gating** | - | - | **87.78% ± 7.9%** |

### Neurophysiological Insights
- **EEG findings**: 48% increase in signal variance for exploratory states
- **NIRS findings**: 42% increase in oxygen consumption during exploration
- **Convergent evidence**: Electrical hyperactivation correlates with metabolic demands

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt                     
├── NIRS_Group_Type_Prediction/
│   ├── notebook/
│   │   └── Prediction_of_Group_NIRS.ipynb
│   ├── NIRS_Data/
│   │   ├── Dataset_overview
│   │   └── extrait.csv
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

### Classification Performance
- **Best unimodal**: EEG with Logistic Regression (81.0% accuracy)
- **Best fusion**: Late Dynamic Gating (87.78% ± 7.9% accuracy)
- **Most stable**: 55.4% variance reduction vs baseline methods

### Feature Importance (SHAP Analysis)
| Feature Type | Top Features | Importance |
|-------------|-------------|------------|
| **EEG** | Signal Variance | 59.2% |
| **EEG** | Spectral Power (Alpha, Theta) | 20.2% |
| **EEG** | RMS Amplitude | 20.2% |
| **NIRS** | Oxygen Consumption | 0.1% |
| **NIRS** | HbO/HbR Levels | 0.1% |

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


*Last updated: January 2025*
