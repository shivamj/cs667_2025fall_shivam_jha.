# Quick Reference Guide - DTW & EAC-DTW

## Core Formulas

### 1. DTW Recurrence Relation
```
D(0,0) = 0
D(i,0) = D(0,j) = ∞  (for i,j > 0)

D(i,j) = d(i,j) + min {
    D(i-1, j),     // insertion
    D(i, j-1),     // deletion
    D(i-1, j-1)    // match
}

where d(i,j) = (q_i - c_j)²
```

### 2. Shannon Entropy
```
H_i = -Σ p_k log₂(p_k)

where p_k = probability of value in bin k
```

### 3. Sigmoid Mapping (Entropy → Window)
```
w_i = w_min + (w_max - w_min) / (1 + exp(-k * (H_i - μ_H)))

where μ_H = mean(H)
```

### 4. Sakoe-Chiba Constraint
```
For row i, compute only columns j where:
    |i - j| ≤ w  (constant window)
```

### 5. EAC-DTW Constraint
```
For row i, compute only columns j where:
    |i - j| ≤ W_i  (adaptive window based on H_i)
```

---

## Python Cheat Sheet

### Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
```

### Compute DTW (from scratch)
```python
def dtw(Q, C):
    n, m = len(Q), len(C)
    DTW = np.full((n+1, m+1), np.inf)
    DTW[0,0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (Q[i-1] - C[j-1])**2
            DTW[i,j] = cost + min(DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1])
    
    return np.sqrt(DTW[n,m])
```

### Compute Entropy
```python
def entropy(signal, window=11, bins=10):
    n = len(signal)
    H = np.zeros(n)
    pad = np.pad(signal, window//2, mode='edge')
    
    for i in range(n):
        segment = pad[i:i+window]
        hist, _ = np.histogram(segment, bins=bins, density=True)
        hist = hist[hist > 0] / hist.sum()
        H[i] = -np.sum(hist * np.log2(hist + 1e-10))
    
    return H
```

### Visualize Cost Matrix
```python
plt.imshow(DTW[1:, 1:], origin='lower', cmap='viridis')
plt.xlabel('Candidate')
plt.ylabel('Query')
plt.colorbar(label='Cumulative Cost')
```

### Visualize Warping Path
```python
path = np.array(warping_path) - 1
plt.plot(path[:,1], path[:,0], 'r-', linewidth=2)
```

---

## Common Mistakes & Fixes

### ❌ Mistake 1: Forget to initialize D(0,0) = 0
```python
# WRONG
DTW = np.zeros((n+1, m+1))

# CORRECT
DTW = np.full((n+1, m+1), np.inf)
DTW[0,0] = 0
```

### ❌ Mistake 2: Use log(0) in entropy
```python
# WRONG
H = -np.sum(hist * np.log2(hist))

# CORRECT
H = -np.sum(hist * np.log2(hist + 1e-10))
```

### ❌ Mistake 3: Forget to filter out zero probabilities
```python
# WRONG
hist, _ = np.histogram(segment, bins=10)

# CORRECT
hist, _ = np.histogram(segment, bins=10, density=True)
hist = hist[hist > 0]  # Remove zeros
```

### ❌ Mistake 4: Index confusion (0-based vs 1-based)
```python
# In DTW matrix: D(i,j) corresponds to Q[i-1] and C[j-1]
cost = (Q[i-1] - C[j-1])**2  # CORRECT
```

---

## Parameter Tuning Guide

### For Standard DTW
- No parameters needed!

### For Sakoe-Chiba DTW
```python
window_percent = 0.10  # 10% of sequence length (conservative)
window_percent = 0.20  # 20% (allows more warping)
```

### For EAC-DTW
```python
w_min = 2          # Tight constraint in low-entropy regions
w_max = 0.15 * n   # 15% of sequence length
k = 2.0            # Moderate transition steepness

# Tuning tips:
# - Increase k for sharper entropy → window mapping
# - Increase w_max if signals have large time shifts
# - Keep w_min small (2-5) to prevent singularities
```

---

## Complexity Cheat Sheet

| Method | Time | Space | Notes |
|--------|------|-------|-------|
| Euclidean | O(n) | O(1) | Fastest, rigid alignment |
| DTW | O(n·m) | O(n·m) | Flexible but slow |
| Sakoe-Chiba DTW | O(n·w) | O(n·m) | Much faster with constraint |
| EAC-DTW | O(n·w_avg) | O(n·m) | Similar to SC-DTW |

---

## Debugging Checklist

When your DTW doesn't work:

- [ ] Are sequences normalized (z-score)?
- [ ] Did you initialize D(0,0) = 0?
- [ ] Did you set boundaries to infinity?
- [ ] Is the cost function squared difference?
- [ ] Are you using correct indices (i-1, j-1)?
- [ ] Did you take sqrt() of final cost?
- [ ] Is the warping path valid (monotonic)?

---

## Quick Tests

### Test 1: Identical Sequences
```python
Q = C = np.array([1, 2, 3, 4, 5])
assert dtw(Q, C) == 0.0  # Should be zero
```

### Test 2: DTW ≤ Euclidean
```python
# DTW should never be worse than Euclidean
assert dtw(Q, C) <= euclidean(Q, C)
```

### Test 3: Entropy Range
```python
H = calculate_entropy(signal)
assert 0 <= H.min() <= H.max() <= np.log2(num_bins)
```

---

## Visualization Template

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Signal comparison
axes[0,0].plot(Q, label='Query')
axes[0,0].plot(C, label='Candidate')
axes[0,0].legend()

# Entropy profile
axes[0,1].plot(H, color='green')
axes[0,1].fill_between(range(len(H)), H, alpha=0.3)

# Cost matrix
axes[1,0].imshow(DTW[1:,1:], origin='lower', cmap='viridis')
axes[1,0].plot(path[:,1], path[:,0], 'r-', linewidth=2)

# Window sizes
axes[1,1].plot(W, color='blue')
axes[1,1].axhline(w_min, color='red', linestyle='--')
axes[1,1].axhline(w_max, color='red', linestyle='--')

plt.tight_layout()
```

---

## One-Liners

### Load sample data
```python
data = np.load('../datasets/sample_ecg_data.npz')
signals, labels = data['signals'], data['labels']
```

### Z-normalize
```python
signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
```

### Add noise at SNR
```python
snr_db = 20
noise_power = np.mean(signal**2) / (10**(snr_db/10))
noisy = signal + np.sqrt(noise_power) * np.random.randn(len(signal))
```

### 1-NN Classification
```python
distances = [dtw(query, candidate) for candidate in train_set]
predicted_label = train_labels[np.argmin(distances)]
```

---

## Resources

- **Documentation**: `help(function_name)` in Python
- **Papers**: See `../papers/PAPERS_INDEX.md`
- **Videos**: See `../tutorials/VIDEO_TUTORIALS.md`
- **Examples**: Check `../code_examples/`

---

**Print this out and keep it handy while coding!** 📋
