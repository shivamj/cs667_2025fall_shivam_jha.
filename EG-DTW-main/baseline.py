"""
Baseline distance metrics for comparison with EG-DTW.
Includes Euclidean distance, standard DTW, and Sakoe-Chiba constrained DTW.
"""

import numpy as np


def euclidean_distance(Q, C):
    """
    Standard Euclidean distance (rigid alignment).
    
    Args:
        Q: Query signal
        C: Candidate signal
    
    Returns:
        distance: Euclidean distance
    """
    if len(Q) != len(C):
        # Resample to same length if needed
        C = np.interp(np.linspace(0, 1, len(Q)), np.linspace(0, 1, len(C)), C)
    return np.sqrt(np.sum((Q - C)**2))


def standard_dtw(Q, C, return_details=False):
    """
    Unconstrained Dynamic Time Warping (maximum elasticity).
    
    WARNING: Susceptible to pathological warping in noisy signals.
    
    Args:
        Q: Query signal
        C: Candidate signal
        return_details: If True, return cost matrix and warping path
    
    Returns:
        distance: DTW distance
        details (optional): Dictionary with cost_matrix and warping_path
    """
    n, m = len(Q), len(C)
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (Q[i-1] - C[j-1]) ** 2
            DTW[i, j] = cost + min(
                DTW[i-1, j],      # Insertion
                DTW[i, j-1],      # Deletion
                DTW[i-1, j-1]     # Match
            )
    
    distance = np.sqrt(DTW[n, m])
    
    if return_details:
        path = _backtrack(DTW, n, m)
        return distance, {'cost_matrix': DTW, 'warping_path': path}
    
    return distance


def sakoe_chiba_dtw(Q, C, window_percent=0.10, return_details=False):
    """
    DTW with fixed Sakoe-Chiba band constraint.
    
    The Sakoe-Chiba band restricts warping to a fixed window around the diagonal.
    
    Args:
        Q: Query signal
        C: Candidate signal
        window_percent: Window size as percentage of signal length (default: 0.10)
        return_details: If True, return cost matrix and warping path
    
    Returns:
        distance: Constrained DTW distance
        details (optional): Dictionary with cost_matrix, warping_path, and window
    """
    n, m = len(Q), len(C)
    window = int(max(n, m) * window_percent)
    
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0
    
    for i in range(1, n + 1):
        # Fixed window constraint
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        
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
        return distance, {
            'cost_matrix': DTW,
            'warping_path': path,
            'window': window
        }
    
    return distance


def _backtrack(DTW, n, m):
    """
    Backtrack through the cost matrix to find the optimal warping path.
    
    Args:
        DTW: Filled cost matrix
        n: Length of query signal
        m: Length of candidate signal
    
    Returns:
        path: Array of (i, j) coordinates representing the warping path
    """
    i, j = n, m
    path = [(i, j)]
    
    while i > 0 and j > 0:
        # Find the direction that led to current cell
        candidates = [
            (i-1, j-1, DTW[i-1, j-1]),  # Diagonal
            (i-1, j, DTW[i-1, j]),      # Vertical
            (i, j-1, DTW[i, j-1])       # Horizontal
        ]
        
        # Choose minimum
        next_i, next_j, _ = min(candidates, key=lambda x: x[2])
        path.append((next_i, next_j))
        i, j = next_i, next_j
    
    # Add remaining path to origin
    while i > 0:
        path.append((i-1, j))
        i -= 1
    while j > 0:
        path.append((i, j-1))
        j -= 1
    
    path.reverse()
    return np.array(path)


if __name__ == "__main__":
    # Test baseline methods
    np.random.seed(42)
    
    signal1 = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    signal2 = np.sin(2 * np.pi * 5 * np.linspace(0.1, 1.1, 100))  # Phase shifted
    
    print("Baseline Distance Metrics Test")
    print("="*50)
    print(f"Euclidean Distance:     {euclidean_distance(signal1, signal2):.4f}")
    print(f"Standard DTW:           {standard_dtw(signal1, signal2):.4f}")
    print(f"Sakoe-Chiba (10%):      {sakoe_chiba_dtw(signal1, signal2, 0.10):.4f}")
    print("="*50)
