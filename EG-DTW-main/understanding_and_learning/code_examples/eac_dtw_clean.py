"""
Entropy-Adaptive Constraint DTW (EAC-DTW) - Complete Implementation
Educational version with detailed comments
"""

import numpy as np

def calculate_entropy(signal, window_size=11, num_bins=10):
    """
    Calculate rolling Shannon entropy profile of a signal.
    
    High entropy = complex/informative region (e.g., ECG QRS complex)
    Low entropy = flat/noisy region (e.g., isoelectric line)
    
    Args:
        signal: Input time series (1D array)
        window_size: Size of sliding window for local entropy
        num_bins: Number of histogram bins for discretization
    
    Returns:
        entropy_profile: Entropy value at each position (same length as signal)
    """
    n = len(signal)
    entropy_profile = np.zeros(n)
    
    # Pad signal to handle boundaries
    pad_size = window_size // 2
    padded_signal = np.pad(signal, (pad_size, pad_size), mode='edge')
    
    for i in range(n):
        # Extract local window
        segment = padded_signal[i:i + window_size]
        
        # Compute histogram (probability distribution)
        hist, _ = np.histogram(segment, bins=num_bins, density=True)
        hist = hist / (hist.sum() + 1e-10)  # Normalize
        hist = hist[hist > 0]  # Remove zero bins
        
        # Shannon entropy: H = -Σ p_k log₂(p_k)
        if len(hist) > 0:
            entropy_profile[i] = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            entropy_profile[i] = 0.0
    
    return entropy_profile


def sigmoid_mapping(entropy_profile, w_min, w_max, k=2.0):
    """
    Map entropy profile to constraint window sizes using sigmoid function.
    
    Formula: w_i = w_min + (w_max - w_min) / (1 + exp(-k * (H_i - μ_H)))
    
    Args:
        entropy_profile: Entropy values (array)
        w_min: Minimum window size (tight constraint for low entropy)
        w_max: Maximum window size (loose constraint for high entropy)
        k: Steepness parameter (higher = sharper transition)
    
    Returns:
        windows: Integer window sizes for each position
    """
    mu_H = np.mean(entropy_profile)  # Mean entropy
    
    # Sigmoid transformation
    sigmoid = 1 / (1 + np.exp(-k * (entropy_profile - mu_H)))
    
    # Map to window range
    windows = w_min + (w_max - w_min) * sigmoid
    
    return np.floor(windows).astype(int)


def eac_dtw_distance(Q, C, w_min=2, w_max_percent=0.15, k=2.0, 
                     return_details=False):
    """
    Entropy-Adaptive Constraint Dynamic Time Warping (EAC-DTW).
    
    Computes DTW with adaptive constraints based on local entropy:
    - Low entropy regions → tight constraint (prevent noise warping)
    - High entropy regions → loose constraint (allow feature alignment)
    
    Args:
        Q: Query sequence (array)
        C: Candidate sequence (array)
        w_min: Minimum window size
        w_max_percent: Maximum window as fraction of sequence length
        k: Sigmoid steepness parameter
        return_details: If True, return entropy, windows, cost matrix, path
    
    Returns:
        distance: EAC-DTW distance
        details (optional): Dict with entropy_profile, window_vector, 
                           cost_matrix, warping_path
    """
    n, m = len(Q), len(C)
    w_max = int(max(n, m) * w_max_percent)
    
    # Step 1: Calculate entropy profile of query sequence
    H = calculate_entropy(Q, window_size=max(10, n//30), num_bins=10)
    
    # Step 2: Map entropy to adaptive window sizes
    W = sigmoid_mapping(H, w_min, w_max, k)
    
    # Step 3: Initialize cost matrix
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0
    
    # Step 4: Fill matrix with entropy-adaptive constraints
    for i in range(1, n + 1):
        # Use window size specific to this position
        w_curr = W[i-1]
        j_start = max(1, i - w_curr)
        j_end = min(m, i + w_curr)
        
        for j in range(j_start, j_end + 1):
            cost = (Q[i-1] - C[j-1]) ** 2
            DTW[i, j] = cost + min(
                DTW[i-1, j],
                DTW[i, j-1],
                DTW[i-1, j-1]
            )
    
    distance = np.sqrt(DTW[n, m])
    
    if return_details:
        path = _backtrack(DTW, n, m)
        details = {
            'entropy_profile': H,
            'window_vector': W,
            'cost_matrix': DTW,
            'warping_path': path,
            'average_window': np.mean(W)
        }
        return distance, details
    
    return distance


def _backtrack(DTW, n, m):
    """Backtrack to find optimal warping path."""
    i, j = n, m
    path = [(i, j)]
    
    while i > 0 and j > 0:
        candidates = [
            (i-1, j-1, DTW[i-1, j-1]),
            (i-1, j, DTW[i-1, j]),
            (i, j-1, DTW[i, j-1])
        ]
        next_i, next_j, _ = min(candidates, key=lambda x: x[2])
        path.append((next_i, next_j))
        i, j = next_i, next_j
    
    while i > 0:
        path.append((i-1, j))
        i -= 1
    while j > 0:
        path.append((i, j-1))
        j -= 1
    
    path.reverse()
    return path


def count_singularities(path):
    """
    Count pathological warping singularities in warping path.
    
    Singularities = runs where i stays same (horizontal) or j stays same (vertical).
    These indicate many-to-one mappings, often caused by noise.
    
    Args:
        path: Warping path as list of (i, j) tuples
    
    Returns:
        count: Number of singularity steps
    """
    singularities = 0
    
    for k in range(1, len(path)):
        i_prev, j_prev = path[k-1]
        i_curr, j_curr = path[k]
        
        # Horizontal singularity (j stays same, i increases)
        if i_curr > i_prev and j_curr == j_prev:
            singularities += 1
        
        # Vertical singularity (i stays same, j increases)
        elif j_curr > j_prev and i_curr == i_prev:
            singularities += 1
    
    return singularities


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Create synthetic ECG-like signal
    t = np.linspace(0, 2, 500)
    Q = np.zeros_like(t)
    
    # Add QRS complex at t=1.0
    Q += 2.0 * np.exp(-200 * (t - 1.0)**2)
    
    # Add noise
    Q += 0.1 * np.random.randn(len(t))
    
    # Create candidate (similar but shifted)
    C = np.zeros_like(t)
    C += 2.0 * np.exp(-200 * (t - 1.1)**2)  # Shifted QRS
    C += 0.1 * np.random.randn(len(t))
    
    # Compute entropy profile
    H = calculate_entropy(Q, window_size=20, num_bins=12)
    W = sigmoid_mapping(H, w_min=2, w_max=30, k=2.0)
    
    print("Entropy-Adaptive Windows:")
    print(f"  Min window: {W.min()}")
    print(f"  Max window: {W.max()}")
    print(f"  Avg window: {W.mean():.2f}")
    
    # Compute EAC-DTW
    distance, details = eac_dtw_distance(Q, C, return_details=True)
    
    print(f"\nEAC-DTW Distance: {distance:.4f}")
    print(f"Singularities: {count_singularities(details['warping_path'])}")
