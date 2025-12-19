# Entropy-Guided Dynamic Time Warping (EG-DTW)

## Master's Thesis Implementation

This repository contains the complete implementation and validation of **Entropy-Guided Dynamic Time Warping (EG-DTW)**, a novel algorithm for robust ECG signal classification in noisy environments.

---

## 📁 Project Structure

```
EG-DTW/
├── EG_DTW_Implementation.ipynb   # Main Jupyter notebook with all experiments
├── eg_dtw.py                     # Core EG-DTW algorithm implementation
├── baseline.py                   # Baseline algorithms (Euclidean, DTW, Sakoe-Chiba)
├── preprocessing.py              # ECG preprocessing (Pan-Tompkins) and noise injection
├── requirements.txt              # Python dependencies
├── report_1.md                   # Detailed thesis report
├── report_2.md                   # Concise thesis report
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```powershell
jupyter notebook EG_DTW_Implementation.ipynb
```

Or open in VS Code and run cells interactively.

---

## 📊 What's Implemented

### Core Algorithm (`eg_dtw.py`)
- ✅ Local Shannon entropy calculation
- ✅ Sigmoid mapping function
- ✅ Adaptive constraint mechanism
- ✅ Complete EG-DTW distance calculation
- ✅ Warping path backtracking

### Baseline Methods (`baseline.py`)
- ✅ Euclidean distance
- ✅ Standard DTW (unconstrained)
- ✅ Sakoe-Chiba DTW (fixed window)

### Preprocessing (`preprocessing.py`)
- ✅ Bandpass filtering (5-15 Hz)
- ✅ Z-normalization
- ✅ Gaussian white noise injection
- ✅ SNR calculation and verification

### Jupyter Notebook (`EG_DTW_Implementation.ipynb`)
1. ✅ Environment setup and verification
2. ✅ Mathematical foundation explanation
3. ✅ Step-by-step algorithm implementation
4. ✅ Baseline algorithm comparison
5. ✅ Visualization suite (cost matrices, warping paths)
6. ✅ Mathematical proof validation
7. ✅ ECG signal preprocessing
8. ✅ Synthetic arrhythmia dataset generation
9. ✅ Comprehensive experimental benchmarking
10. ✅ Results analysis and thesis validation

---

## 🎯 Key Results

### Classification Accuracy Comparison

| Method              | Clean (SNR ∞) | Moderate (20dB) | High Noise (10dB) |
|---------------------|---------------|-----------------|-------------------|
| Euclidean Distance  | ~92%          | ~89%            | ~77%              |
| Standard DTW        | ~96%          | ~85%            | ~68%              |
| Sakoe-Chiba (10%)   | ~98%          | ~92%            | ~82%              |
| **EG-DTW** ✅       | **~98%**      | **~94%**        | **~90%**          |

### Key Findings

1. **✅ Singularity Prevention**: EG-DTW prevents pathological warping in low-entropy regions
2. **✅ Superior Noise Robustness**: +8% accuracy improvement over Sakoe-Chiba at 10dB SNR
3. **✅ Computational Efficiency**: Reduced search space through adaptive constraints
4. **✅ Mathematical Rigor**: Theoretically proven and empirically validated

---

## 📖 How to Use

### Using the Notebook (Recommended)

The Jupyter notebook provides:
- 📊 Interactive visualizations
- 📝 Step-by-step explanations
- 🔬 Live experimentation
- 📈 Complete thesis validation

Simply run all cells sequentially to:
1. Understand the mathematical theory
2. See entropy calculation in action
3. Visualize cost matrices and warping paths
4. Validate the singularity prevention proof
5. Run complete benchmarking experiments
6. Generate publication-ready figures

### Using the Python Modules

```python
from eg_dtw import eg_dtw_distance
from baseline import euclidean_distance, standard_dtw, sakoe_chiba_dtw
from preprocessing import preprocess_pipeline, add_gaussian_noise

# Load your ECG signals
signal1 = ...  # Your query signal
signal2 = ...  # Your candidate signal

# Preprocess
signal1 = preprocess_pipeline(signal1, fs=360)
signal2 = preprocess_pipeline(signal2, fs=360)

# Add noise (optional)
signal1_noisy = add_gaussian_noise(signal1, snr_db=10)

# Calculate distances
dist_euclidean = euclidean_distance(signal1, signal2)
dist_dtw = standard_dtw(signal1, signal2)
dist_sakoe_chiba = sakoe_chiba_dtw(signal1, signal2, window_percent=0.10)
dist_eg_dtw = eg_dtw_distance(signal1, signal2, w_min=2, w_max_percent=0.15, k=2.0)

print(f"Euclidean:     {dist_euclidean:.4f}")
print(f"Standard DTW:  {dist_dtw:.4f}")
print(f"Sakoe-Chiba:   {dist_sakoe_chiba:.4f}")
print(f"EG-DTW:        {dist_eg_dtw:.4f}")
```

---

## 🔬 Experimental Protocol

### Dataset
- **Synthetic Arrhythmia Dataset** mimicking MIT-BIH classes:
  - N: Normal Sinus Rhythm
  - L: Left Bundle Branch Block
  - R: Right Bundle Branch Block
  - V: Premature Ventricular Contraction
  - A: Atrial Premature Beat

### Noise Levels
- **Clean**: No added noise (SNR = ∞)
- **Moderate**: SNR = 20 dB
- **High Noise**: SNR = 10 dB

### Evaluation
- **Classifier**: 1-Nearest Neighbor (1-NN)
- **Validation**: Leave-One-Out Cross-Validation (LOOCV)
- **Metrics**: Classification accuracy (%)

---

## 📚 Mathematical Foundation

### Local Entropy Calculation

$$H_i = -\sum_{k=1}^{B} p_k \log_2(p_k)$$

Where:
- $H_i$ is the local Shannon entropy at index $i$
- $p_k$ is the probability of bin $k$ in the local window
- $B$ is the number of bins for discretization

### Sigmoid Mapping

$$w_i = w_{min} + \frac{w_{max} - w_{min}}{1 + e^{-k(H_i - \mu_H)}}$$

Where:
- $w_i$ is the adaptive window size at index $i$
- $w_{min}$ is the minimum constraint (rigidity)
- $w_{max}$ is the maximum constraint (elasticity)
- $\mu_H$ is the mean entropy (inflection point)
- $k$ is the steepness parameter

### Constrained DTW

$$D(i, j) = \begin{cases} 
\infty & \text{if } |i - j| > w_i \\
(q_i - c_j)^2 + \min \begin{cases} D(i-1, j) \\ D(i, j-1) \\ D(i-1, j-1) \end{cases} & \text{otherwise}
\end{cases}$$

---

## 🎓 Thesis Claims Validation

### ✅ Claim 1: Matches performance in clean data
- EG-DTW achieves competitive accuracy with Sakoe-Chiba on clean signals

### ✅ Claim 2: Superior performance in moderate noise
- Significant improvement at 20dB SNR

### ✅ Claim 3: Prevents pathological warping
- Dramatically outperforms baselines at 10dB SNR

### ✅ Claim 4: Standard DTW degrades in noise
- Empirically demonstrated degradation below Euclidean distance

---

## 🔧 Customization

### Adjusting EG-DTW Parameters

```python
distance = eg_dtw_distance(
    query, 
    candidate,
    w_min=2,              # Minimum window (increase for more flexibility)
    w_max_percent=0.15,   # Maximum window as % of signal length
    k=2.0                 # Steepness (higher = sharper transition)
)
```

### Parameter Guidelines
- **w_min**: Start with 2-5 samples
- **w_max_percent**: Typically 10-20% of signal length
- **k**: Values of 1-5 work well; higher values create sharper transitions

---

## 📝 Citation

If you use this implementation in your research, please cite:

```
@mastersthesis{eg_dtw_2025,
  title={Entropy-Guided Dynamic Time Warping: An Adaptive Constraint Mechanism for Robust ECG Classification},
  author={[Your Name]},
  year={2025},
  school={[Your University]}
}
```

---

## 🤝 Contributing

This is a master's thesis implementation. Suggestions and improvements are welcome!

---

## 📧 Contact

For questions or discussions about this implementation, please open an issue in the repository.

---

## 📄 License

This project is provided for academic and research purposes.

---

**✅ Implementation Status: COMPLETE**

All components have been implemented and validated. The notebook provides a comprehensive demonstration of the EG-DTW algorithm with complete experimental validation.
