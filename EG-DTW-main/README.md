# Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW)
## An Entropy-Driven Alignment Strategy for Robust ECG Classification

**Authors:** Fnu Ashutosh, Shivam Jha  
**Faculty Advisor:** Sung-Hyuk Cha  
**Institution:** Seidenberg School of Computer Science and Information Systems, Pace University  
**Date:** December 5, 2025  
**Event:** Seidenberg Annual Research Day 2025  

---

## 1. Executive Summary
Unconstrained Dynamic Time Warping (DTW) suffers from "pathological warping" in noisy signals, often aligning transient noise artifacts with significant clinical features. We propose **EAC-DTW**, a novel approach that uses local Shannon entropy to dynamically adjust the warping constraint window. By tightening constraints in low-entropy (noisy/flat) regions and loosening them in high-entropy (QRS complex) regions, we achieve robust alignment without the rigidity of fixed-band methods.

---

## 2. Mathematical Formulation

### 2.1 Local Shannon Entropy ($H_i$)
To quantify the "informativeness" of the signal at time step $i$, we calculate the rolling Shannon entropy over a local window. High entropy indicates complex morphology (e.g., QRS), while low entropy indicates noise or isoelectric lines.

$$ H_i = -\sum_{k=1}^{B} p_k \log_2(p_k) $$

*   $p_k$: Probability density of signal amplitudes in the local window.
*   $B$: Number of histogram bins.

### 2.2 Adaptive Constraint Mapping ($w_i$)
We map the entropy profile to a dynamic window size $w_i$ using a Sigmoid function. This ensures a smooth transition between rigid constraints (for noise) and elastic constraints (for features).

$$ w_i = w_{min} + \frac{w_{max} - w_{min}}{1 + e^{-k(H_i - \mu_H)}} $$

*   **$w_{min}$**: Minimum band radius (suppresses noise matching).
*   **$w_{max}$**: Maximum band radius (allows feature alignment).
*   **$k$**: Steepness parameter (sensitivity).
*   **$\mu_H$**: Mean entropy of the signal.

### 2.3 The Constrained Distance Calculation
The optimal warping path minimizes the cumulative cost $D(i, j)$, subject to the dynamic constraint:

$$ D(i, j) = (x_i - y_j)^2 + \min(D(i-1, j), D(i, j-1), D(i-1, j-1)) $$

**Constraint:** The search is strictly limited to:
$$ |i - j| \le w_i $$

---

## 3. Algorithm Implementation

The core logic shifts from an $O(N^2)$ global search to an $O(N \cdot \bar{w})$ constrained search.

```python
def EAC_DTW(Query, Candidate):
    """
    Entropy-Adaptive Constraint DTW Implementation Logic
    """
    # 1. Calculate Entropy Profile
    # High entropy = QRS complex; Low entropy = Flat/Noise
    H = calculate_rolling_entropy(Query, window_size=10)
    
    # 2. Map to Dynamic Constraints
    # Returns a vector of window sizes, not a single scalar
    W_dynamic = sigmoid_mapping(H, w_min=2, w_max=30)
    
    # 3. Dynamic Programming with Adaptive Band
    n, m = len(Query), len(Candidate)
    DTW = initialize_matrix(n, m)
    
    for i in range(1, n):
        # The constraint width 'w' changes at every time step 'i'
        w = W_dynamic[i]
        
        # Define the "Tunnel" (Valid search area)
        j_start = max(0, i - w)
        j_end   = min(m, i + w)
        
        for j in range(j_start, j_end):
            cost = euclidean_dist(Query[i], Candidate[j])
            DTW[i,j] = cost + min_neighbor(DTW, i, j)
            
    return sqrt(DTW[n,m])