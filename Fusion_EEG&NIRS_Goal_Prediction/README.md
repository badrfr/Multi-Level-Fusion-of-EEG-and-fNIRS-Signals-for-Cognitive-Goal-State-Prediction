# EEG-NIRS Multimodal Fusion for Cognitive Load Classification

A comprehensive machine learning framework for classifying cognitive states using multimodal EEG-NIRS data fusion. This project implements and compares three fusion strategies to distinguish between with objective (AO) and without objective (SO) cognitive states.
---
## 🧠 Overview

This repository contains the implementation of a multimodal neurophysiological signal classification system that combines electroencephalography (EEG) and near-infrared spectroscopy (NIRS) data to predict cognitive load states. The project explores three distinct fusion approaches with advanced machine learning techniques.

### Key Features

- **Three Fusion Strategies**: Early, intermediate, and late fusion approaches
- **Advanced Dynamic Gating**: Instance-specific adaptive fusion with meta-learning
- **Comprehensive Model Comparison**: 9 different ML/DL models evaluated
- **SHAP Interpretability**: Feature importance analysis and model explainability
- **Robust Evaluation**: 5-fold cross-validation with statistical analysis

---

## 📊 Results Summary

| Fusion Strategy | Best Model | Accuracy | F1-Score | ROC-AUC |
|----------------|------------|----------|----------|---------|
| **Early Fusion** | SVM | 66.11% ± 7.7% | 0.690 ± 0.089 | 0.828 ± 0.104 |
| **Intermediate Fusion** | Logistic Regression | 70.6% ± 10.3% | 0.697 ± 0.183 | 0.637 ± 0.139 |
| **Late Fusion** | Dynamic Gating | **87.78% ± 7.9%** | **0.874 ± 0.085** | **0.903 ± 0.110** |

---

## 📁 Project Structure

```
Fusion_EEG&NIRS_Group_Type_Prediction/
  ├── Early_Fusion/
  │     ├── notebooks/
  │     │    ├── Early_fusion.ipynb
  │     │    └── EEG_NIRS_Construct.ipynb
  │     └── Fusion_Data
  │          ├── Dataset_overview
  │          └── extrait.csv       
  ├── Features_Fusion/
  │     └── Features_fusion.ipynb     
  ├── Late_Fusion/
  │     └── Late_fusion.ipynb
  └── README.md

```
---

## 🔬 Methodology

### 1. Early Fusion (Signal-Level Integration)

- **Data Synchronization**: EEG downsampled from 512Hz to 4Hz to match NIRS
- **Temporal Alignment**: Sample-by-sample correspondence
- **Concatenation**: Direct signal concatenation into unified matrix
- **Best Performance**: SVM with 66.11% accuracy

### 2. Intermediate Fusion (Feature-Level Merging)

- **Separate Feature Extraction**: Domain-specific features for each modality
- **Feature Concatenation**: Combined multimodal feature vectors
- **Classification**: Various ML models on fused features
- **Best Performance**: Logistic Regression with 70.6% accuracy

### 3. Late Fusion (Decision-Level Combination)

Advanced multi-stage approach with three phases:

#### Phase 1: Baseline Weighted Fusion
- Performance-proportional weight assignment
- Accuracy: 75.56% ± 17.7%

#### Phase 2: Advanced Strategies
- 4 weighting methods × 5 fusion strategies = 20 configurations
- Dynamic confidence adaptation
- Accuracy: 77.78% ± 18.5%

#### Phase 3: Dynamic Gating
- **Meta-Feature Engineering**: 11-dimensional feature space
- **Instance-Specific Weights**: Gradient Boosting Regressor
- **Optimal Weight Learning**: Exhaustive search + supervised learning
- **Best Performance**: 87.78% ± 7.9% accuracy

---

## 🎯 Advanced Features

### Dynamic Gating Meta-Features

The dynamic gating model uses 11 meta-features:
- Raw probabilities (P_EEG, P_NIRS)
- Logit transformations
- Confidence scores
- Binary entropies
- Inter-modality disagreement
- Geometric and harmonic means

### SHAP Interpretability

- **EEG Importance**: Alpha, theta, delta bands, entropy, RMS
- **NIRS Importance**: Ultra-low power indices, mean hemodynamic features
- **Modality Weighting**: EEG (65% ± 7.6%) vs NIRS (35% ± 7.6%)

---

## 📈 Performance Analysis

### Key Achievements

- **87.78% Accuracy**: Best-in-class performance with dynamic gating
- **55.4% Variance Reduction**: Improved stability and reliability
- **90.25% ROC-AUC**: Superior discrimination capability
- **Complementary Information**: EEG and NIRS provide distinct cognitive markers

### Model Comparison

| Model | Early Fusion | Intermediate Fusion | Late Fusion |
|-------|-------------|-------------------|-------------|
| Traditional ML | ✅ Best | ✅ Best | ✅ Best |
| Deep Learning | ❌ Overfitting | ❌ Inconsistent | ⚠️ Moderate |
| Ensemble Methods | ⚠️ Moderate | ⚠️ Good | ✅ Excellent |

---

## Pipeline Complet d'Amélioration de l'Accuracy

```mermaid
flowchart LR
    %% Data Input
    A1[Raw EEG Data<br/>512Hz] --> A2[EEG Preprocessing<br/>Downsampling to 4Hz]
    B1[Raw NIRS Data<br/>4Hz] --> B2[NIRS Preprocessing<br/>Artifact Removal]

    %% Synchronization
    A2 --> C[Temporal Alignment<br/>Sample-by-sample]
    B2 --> C

    %% Three Fusion Approaches
    C --> D{Three Fusion<br/>Strategies}

    %% Early Fusion Branch
    D --> E1[Early Fusion<br/>Signal-Level]
    E1 --> E2[Direct Signal<br/>Concatenation]
    E2 --> E3[SVM Classifier]
    E3 --> E4[Result: 66.11%<br/>± 7.7%]

    %% Intermediate Fusion Branch
    D --> F1[Intermediate Fusion<br/>Feature-Level]
    F1 --> F2[EEG Feature<br/>Extraction]
    F1 --> F3[NIRS Feature<br/>Extraction]
    F2 --> F4[Feature<br/>Concatenation]
    F3 --> F4
    F4 --> F5[Logistic Regression]
    F5 --> F6[Result: 70.6%<br/>± 10.3%]

    %% Late Fusion Branch
    D --> G1[Late Fusion<br/>Decision-Level]

    %% Late Fusion Phase 1
    G1 --> H1[Phase 1:<br/>Baseline Weighted]
    H1 --> H2[Performance-Based<br/>Weights]
    H2 --> H3[Linear Combination]
    H3 --> H4[Result: 75.56%<br/>± 17.7%]

    %% Late Fusion Phase 2
    G1 --> I1[Phase 2:<br/>Advanced Strategies]
    I1 --> I2[4 Weighting Methods<br/>× 5 Fusion Strategies]
    I2 --> I3[20 Configurations<br/>Tested]
    I3 --> I4[Best: Standard +<br/>Dynamic Confidence]
    I4 --> I5[Result: 77.78%<br/>± 18.5%]

    %% Late Fusion Phase 3 - Dynamic Gating
    G1 --> J1[Phase 3:<br/>Dynamic Gating]
    J1 --> J2[Oracle Weight<br/>Learning]
    J1 --> J3[Meta-Feature<br/>Engineering 11D]
    J2 --> J4[Gradient Boosting<br/>Gating Model]
    J3 --> J4
    J4 --> J5[Instance-Specific<br/>Weight Prediction]
    J5 --> J6[Result: 87.78%<br/>± 7.9%]

    %% Additional Methods
    J1 --> K1[Logit Stacking<br/>87.50% ± 7.9%]
    J1 --> K2[Enhanced DynConf<br/>83.06% ± 12.5%]

    %% Final Analysis
    J6 --> L1[SHAP Analysis<br/>Interpretability]
    K1 --> L1
    K2 --> L1

    L1 --> L2[EEG Dominance<br/>65% ± 7.6%]
    L1 --> L3[NIRS Support<br/>35% ± 7.6%]
    L1 --> L4[Meta-Features<br/>Importance]

    %% Performance Summary
    E4 --> M1[Performance<br/>Comparison]
    F6 --> M1
    H4 --> M1
    I5 --> M1
    J6 --> M1

    M1 --> M2[🎯 Total Gain:<br/>+21.67 points<br/>55.4% variance reduction]

    %% Softer Styling
    classDef earlyFusion fill:#fdecea,stroke:#d32f2f,stroke-width:2px,color:#000
    classDef intermediateFusion fill:#e9f7ef,stroke:#388e3c,stroke-width:2px,color:#000
    classDef lateFusion fill:#eaf2fb,stroke:#1976d2,stroke-width:2px,color:#000
    classDef breakthrough fill:#fff7e6,stroke:#f57c00,stroke-width:3px,color:#000
    classDef result fill:#f9f0fa,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef final fill:#e3f8ff,stroke:#0277bd,stroke-width:3px,color:#000

    class E1,E2,E3,E4 earlyFusion
    class F1,F2,F3,F4,F5,F6 intermediateFusion
    class G1,H1,H2,H3,H4,I1,I2,I3,I4,I5 lateFusion
    class J1,J2,J3,J4,J5,J6,K1,K2 breakthrough
    class L1,L2,L3,L4 result
    class M1,M2 final
```
---

## Méthodologie Détaillée

### Phase 1: Approches de Base
```mermaid
flowchart TD
    A[EEG-NIRS Data<br/>41 paired samples] --> B{Fusion Strategy}
    
    B --> C[Early Fusion<br/>66.11% ± 7.7%]
    B --> D[Intermediate Fusion<br/>70.6% ± 10.3%]
    B --> E[Late Fusion Baseline<br/>75.56% ± 17.7%]
    
    C --> F[❌ High variance<br/>❌ Signal interference]
    D --> G[✅ Better than early<br/>❌ Still unstable]
    E --> H[✅ Promising direction<br/>❌ Very high variance]
    
    F --> I[💡 Need better integration]
    G --> I
    H --> I
    
    I --> J[Advanced Late Fusion<br/>Strategies]
```

### Phase 2: Optimisation Avancée
```mermaid
flowchart LR
    A[Late Fusion Optimization] --> B[Systematic exploration<br/>4 x 5 = 20 configurations]
    B --> C[Weighting methods]
    B --> D[Fusion strategies]

    C --> C1["Standard<br/>(performance-based)"]
    C --> C2["Performance-Confidence<br/>(ACC x confidence)"]
    C --> C3["Exponential<br/>exp(a * ACC)"]
    C --> C4["Softmax<br/>temperature-scaled"]

    D --> D1["Weighted average<br/>(linear combination)"]
    D --> D2["Dynamic confidence<br/>(instance adaptation)"]
    D --> D3["Geometric mean<br/>(non-linear fusion)"]
    D --> D4["Adaptive threshold<br/>(variable boundary)"]
    D --> D5["Max confidence<br/>(winner-take-all)"]

    C1 --> E["Best observed result<br/>Standard + Dynamic Confidence<br/>77.78% ± 18.5%"]
    C2 --> E
    C3 --> E
    C4 --> E
    D1 --> E
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E

    E --> F["Marginal improvement; variance still high<br/>⇒ Need paradigm shift"]
```

### Phase 3: Révolution Dynamic Gating
```mermaid
flowchart TD
    A[Dynamic Gating — Paradigm shift] --> B["Oracle weight learning<br/>(exhaustive search on 51 points)"]
    A --> C["Meta-feature engineering<br/>(11-dimensional)"]

    B --> D["Optimal weights per training sample"]

    C --> E["Raw probabilities<br/>P_EEG, P_NIRS"]
    C --> F["Logit transforms<br/>log(p/(1-p))"]
    C --> G["Confidence scores<br/>abs(p - 0.5) * 2"]
    C --> H["Binary entropies<br/>-p*log(p) - (1-p)*log(1-p)"]
    C --> I["Disagreement<br/>abs(P_EEG - P_NIRS)"]
    C --> J["Geometric mean<br/>sqrt(P_EEG * P_NIRS)"]
    C --> K["Harmonic mean<br/>2*P_EEG*P_NIRS/(P_EEG + P_NIRS)"]

    D --> L["Gradient Boosting Regressor training"]
    E --> L
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M["Gating model<br/>n_estimators=200, lr=0.05, depth=3"]
    M --> N["Instance-specific weight prediction"]

    N --> O["Accuracy 87.78% ± 7.9%<br/>+10% vs Phase 2<br/>Variance -55.4%"]
```

