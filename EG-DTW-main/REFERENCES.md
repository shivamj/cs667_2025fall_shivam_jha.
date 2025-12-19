# References Used in EAC-DTW Implementation

## Core Dynamic Time Warping References

### 1. **Sakoe & Chiba (1978)** - Foundation of DTW Constraints
**Full Citation:**
> Sakoe, H., & Chiba, S. (1978). "Dynamic programming algorithm optimization for spoken word recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.

**Used In:**
- Baseline `sakoe_chiba_dtw()` function (Cell #12)
- Mathematical foundation section describing Sakoe-Chiba band constraints (Cell #4)
- Abstract positioning EAC-DTW as extension of Sakoe-Chiba approach

**Relevance:**
Introduced the concept of global path constraints (Sakoe-Chiba band) to limit DTW warping flexibility. EAC-DTW extends this by making the band width adaptive rather than fixed.

---

### 2. **Müller (2007)** - Information Retrieval for Music and Motion
**Full Citation:**
> Müller, M. (2007). *Information Retrieval for Music and Motion*. Springer.

**Used In:**
- Baseline `standard_dtw()` function documentation (Cell #12)

**Relevance:**
Comprehensive reference on DTW implementation details and applications in time series analysis.

---

### 3. **Ratanamahatana & Keogh (2004)** - DTW Critical Analysis
**Full Citation:**
> Ratanamahatana, C. A., & Keogh, E. (2004). "Everything you know about Dynamic Time Warping is wrong." *Third Workshop on Mining Temporal and Sequential Data*, pp. 22-25.

**Used In:**
- Baseline `sakoe_chiba_dtw()` function documentation (Cell #12)

**Relevance:**
Critical analysis of common DTW misconceptions and best practices for constraint selection.

---

### 4. **Cuturi (2017)** - Soft-DTW (Differentiation)
**Full Citation:**
> Cuturi, M., & Blondel, M. (2017). "Soft-DTW: a differentiable loss function for time-series." *International Conference on Machine Learning*, pp. 894-903.

**Used In:**
- Mathematical Foundation section (Cell #4) - explicitly distinguishing EAC-DTW from Soft-DTW
- Conclusions section positioning EAC-DTW in literature

**Relevance:**
**Important Distinction:** Soft-DTW uses differentiable smoothing via Gibbs distributions for gradient-based optimization. EAC-DTW takes a different approach using entropy-driven constraint adaptation without altering the fundamental dynamic programming recurrence.

---

## Signal Processing & ECG References

### 5. **Pan & Tompkins (1985)** - ECG QRS Detection
**Full Citation:**
> Pan, J., & Tompkins, W. J. (1985). "A Real-Time QRS Detection Algorithm." *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230-236.

**Used In:**
- `bandpass_filter()` function in preprocessing module (Cell #20)
- Justification for 5-15 Hz frequency range for QRS preservation

**Relevance:**
Standard reference for ECG preprocessing. The 5-15 Hz bandpass filter design preserves QRS complex dominant energy while suppressing baseline wander and high-frequency noise.

---

## Dataset References

### 6. **MIT-BIH Arrhythmia Database** (Mentioned, Not Used)
**Full Citation:**
> Moody, G. B., & Mark, R. G. (2001). "The impact of the MIT-BIH Arrhythmia Database." *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.

**Used In:**
- Abstract and methodology sections as **future work** benchmark
- Explicitly noted as NOT used in current implementation (synthetic data only)

**Access:**
```python
import wfdb
record = wfdb.rdrecord('mitdb/100')
```

**Relevance:**
Gold standard for ECG arrhythmia classification research. Current work uses synthetic data; MIT-BIH validation is planned for clinical deployment assessment.

---

## Additional Methods Mentioned (Not Primary References)

### FastDTW (Salvador & Chan 2007)
**Citation:**
> Salvador, S., & Chan, P. (2007). "FastDTW: Toward accurate dynamic time warping in linear time and space." *Intelligent Data Analysis*, 11(5), 561-580.

**Mentioned In:** Conclusions section as related approach using multi-resolution coarsening

### Derivative DTW
**Mentioned In:** Conclusions section as alternative using first derivatives rather than amplitude constraints

---

## Mathematical Foundations

### Shannon Entropy
**Concept Used:**
$$H_i = -\sum_{k=1}^{B} p_k \log_2(p_k)$$

**Standard Information Theory Reference:**
> Shannon, C. E. (1948). "A mathematical theory of communication." *Bell System Technical Journal*, 27(3), 379-423.

**Implementation:** `calculate_entropy()` function using histogram-based probability estimation

---

## Summary Statistics

**Total Direct Citations:** 6 primary references
- **DTW Methods:** 4 (Sakoe & Chiba, Müller, Ratanamahatana & Keogh, Cuturi)
- **Signal Processing:** 1 (Pan & Tompkins)
- **Dataset:** 1 (MIT-BIH - future work)

**Citation Style:** IEEE format used throughout implementation

**Code Attribution:** All baseline algorithms properly attributed to original sources in docstrings

---

## How to Cite This Implementation

### APA Format:
```
Ashutosh, F., & Cha, S. (2025). Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW): 
An Entropy-Driven Alignment Strategy for Robust ECG Classification. Seidenberg Annual Research Day, 
Pace University, New York, NY.
```

### BibTeX:
```bibtex
@inproceedings{ashutosh2025eacdtw,
  title={Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW): An Entropy-Driven Alignment Strategy for Robust ECG Classification},
  author={Ashutosh, Fnu and Cha, Sung-Hyuk},
  booktitle={Seidenberg Annual Research Day},
  year={2025},
  organization={Pace University}
}
```

---
