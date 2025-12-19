Seidenberg Annual Research Day 2025 (December 5, 2025)

Title: Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW): An Entropy-Driven Alignment Strategy for Robust ECG Classification

Authors: Fnu Ashutosh and Sung-Hyuk Cha
Affiliation: Seidenberg School of Computer Science and Information Systems, Pace University, New York, NY, USA
Emails: {an05893n, scha}@pace.edu

Abstract:
Dynamic Time Warping (DTW) is widely used for temporal alignment in physiological signal analysis, yet unconstrained DTW can suffer from pathological warping in noisy segments—aligning transient artifacts with clinically meaningful morphology (e.g., QRS complexes). Fixed global constraints such as the Sakoe-Chiba band reduce excessive elasticity but cannot adapt to heterogeneous structure in Electrocardiogram (ECG) signals that alternate between high-complexity (QRS) and low-complexity (isoelectric) regions. We present Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW), a modified DTW formulation that computes a rolling Shannon entropy profile and maps it through a sigmoid to produce a position-dependent constraint vector. Low-entropy regions receive tight warping limits to suppress singularities; high-entropy regions allow broader alignment flexibility to preserve morphological fidelity. Using a controlled synthetic ECG-like dataset (five arrhythmia classes: Normal, LBBB, RBBB, PVC, APC) under three noise conditions (clean, 20 dB, 10 dB SNR), EAC-DTW achieves 79.3% classification accuracy at 10 dB—improving by 6.0 percentage points over a fixed 10% Sakoe-Chiba band and outperforming unconstrained DTW in noise robustness. Singularity counts (horizontal/vertical path runs) are reduced (168 vs 286 for standard DTW), indicating mitigation of pathological warping. These results, while promising, are preliminary: clinical validation on the MIT-BIH Arrhythmia Database is planned to assess generalizability. The contribution is incremental—adapting classical band constraints via entropy rather than proposing a differentiable relaxation (e.g., Soft-DTW). Future work includes parameter sensitivity optimization (k, window bounds), real-data benchmarking, multilead extensions, and potential integration with learned feature representations.

Keywords: Dynamic Time Warping, ECG Classification, Adaptive Constraints, Shannon Entropy, Time Series Alignment, Noise Robustness

---

Condensed Abstract (≈245 words):
Dynamic Time Warping (DTW) enables elastic alignment of temporal signals but its unconstrained form is prone to pathological warping in noisy physiological data—allowing transient artifacts to align with clinically meaningful morphology (e.g., QRS complexes). Fixed global constraints such as the Sakoe-Chiba band reduce excessive flexibility yet cannot adapt to heterogeneous Electrocardiogram (ECG) structure that alternates between high-complexity and flat isoelectric regions. We present Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW), an incremental modification that derives a rolling Shannon entropy profile and maps it through a sigmoid to produce a position-dependent constraint vector. Low-entropy regions receive tight local bounds to suppress singularities while high-entropy regions retain alignment elasticity to preserve morphological fidelity. Using a controlled synthetic ECG-like dataset spanning five arrhythmia classes (Normal, LBBB, RBBB, PVC, APC) under clean, 20 dB, and 10 dB SNR conditions, EAC-DTW attains 79.3% accuracy at 10 dB—improving noise robustness by 6.0 percentage points over a fixed 10% Sakoe-Chiba band and outperforming unconstrained DTW. Warping path singularities (horizontal/vertical runs) are reduced (168 vs 286 for standard DTW), indicating effective mitigation of pathological deviations. These findings are preliminary: clinical validation on the MIT-BIH Arrhythmia Database is required to assess real-world generalizability. Unlike differentiable relaxations (e.g., Soft-DTW), EAC-DTW adaptively reshapes classical band constraints via entropy without altering the fundamental dynamic programming recurrence. Future work will examine parameter sensitivity (k, window limits), multilead extensions, and integration with learned representations for hybrid entropy-guided alignment.

Keywords (Condensed Version): Dynamic Time Warping; ECG; Adaptive Constraints; Entropy; Noise Robustness

---

## References

[1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 26, no. 1, pp. 43-49, 1978.

[2] M. Müller, *Information Retrieval for Music and Motion*, Springer, New York, 2007.

[3] C. A. Ratanamahatana and E. Keogh, "Everything you know about Dynamic Time Warping is wrong," in *Third Workshop on Mining Temporal and Sequential Data*, pp. 22-25, 2004.

[4] M. Cuturi and M. Blondel, "Soft-DTW: a differentiable loss function for time-series," in *International Conference on Machine Learning*, pp. 894-903, 2017.

[5] J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm," *IEEE Transactions on Biomedical Engineering*, vol. BME-32, no. 3, pp. 230-236, 1985.

[6] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45-50, 2001.

[7] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 3, pp. 379-423, 1948.

---

## Abstract Validation Summary

**Numerical Claims Verified Against Notebook Results:**
✅ 79.3% accuracy at 10 dB SNR (EG-DTW) - CONFIRMED (line 1467, 1569)
✅ 6.0 percentage point improvement over Sakoe-Chiba (79.3% - 73.3%) - CONFIRMED
✅ 168 singularities for EG-DTW vs 286 for Standard DTW - CONFIRMED (lines 672, 775, 787)
✅ Three SNR levels tested: clean (∞), 20 dB, 10 dB - CONFIRMED
✅ Five arrhythmia classes: N, L, R, V, A (Normal, LBBB, RBBB, PVC, APC) - CONFIRMED
✅ Synthetic ECG-like dataset - CONFIRMED and properly disclosed

**Terminology Issue Identified:**
⚠️ Notebook uses "EG-DTW" throughout (Entropy-Guided)
⚠️ Abstract uses "EAC-DTW" (Entropy-Adaptive Constraint)
📌 RECOMMENDATION: Use "EAC-DTW" in abstract AND update notebook to maintain consistency

---

## Recommended Figures for SARD Poster/Presentation

### **Figure 1: Algorithm Mechanism (Cell #VSC-41a4334d, #VSC-108f6d56)**
**Title:** "Entropy-Adaptive Window Sizing Mechanism"
**Description:** Two-panel figure showing:
- Top: Synthetic ECG signal with entropy profile overlay
- Bottom: Sigmoid mapping transforming entropy to adaptive window sizes
**Impact:** Demonstrates the core innovation—how local signal complexity drives constraint adaptation
**Key Insight:** QRS peaks (high entropy) → wide windows; flat segments (low entropy) → tight constraints

### **Figure 2: Singularity Reduction (Cell #VSC-02860549)**
**Title:** "Pathological Warping Mitigation Analysis"
**Description:** Three-column comparison (Standard DTW, Sakoe-Chiba, EAC-DTW) showing:
- Top row: Cost matrices with warping paths overlaid
- Bottom row: Singularity counts highlighted on paths
**Impact:** Visual proof that EAC-DTW reduces pathological alignments (168 vs 286 singularities)
**Key Insight:** Adaptive constraints prevent noise-to-feature misalignments

### **Figure 3: Classification Performance (Cell #VSC-98cd7892)**
**Title:** "Noise Robustness Comparison Across SNR Levels"
**Description:** Dual visualization:
- Left: Grouped bar chart (4 methods × 3 SNR conditions)
- Right: Line plot showing degradation curves under noise
**Impact:** Demonstrates practical value—6.0pp accuracy gain at 10 dB SNR
**Key Insight:** EAC-DTW maintains performance under high noise where Standard DTW degrades below Euclidean baseline

### **Figure 4 (Optional): Parameter Sensitivity (Cell #VSC-0f685342)**
**Title:** "Sigmoid Steepness Parameter (k) Optimization"
**Description:** Line plot showing accuracy vs k values (0.5 to 5.0)
**Impact:** Shows due diligence in parameter selection and sensitivity analysis
**Key Insight:** Performance plateaus at k≥2.5; k=2.0 selected as conservative baseline

---

## Export Recommendations

For high-quality poster graphics:
```python
# In notebook, add before plt.show():
plt.savefig('figure1_mechanism.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_mechanism.pdf', bbox_inches='tight')  # Vector format
```

**Optimal Figure Selection for Single-Panel Abstract:**
If limited to ONE figure, use **Figure 3** (Performance comparison) - directly supports the 79.3% claim and shows practical impact.
