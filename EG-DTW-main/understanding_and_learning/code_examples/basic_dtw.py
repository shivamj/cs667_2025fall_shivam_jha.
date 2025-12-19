"""
Basic DTW Implementation - Educational Version
Step-by-step DTW from scratch (no libraries)
"""

import numpy as np

def euclidean_distance(q, c):
    """
    Simple Euclidean distance between two equal-length sequences.
    
    Args:
        q, c: Input sequences (arrays)
    
    Returns:
        Euclidean distance (float)
    """
    if len(q) != len(c):
        # Resample to same length
        c = np.interp(np.linspace(0, 1, len(q)), np.linspace(0, 1, len(c)), c)
    return np.sqrt(np.sum((q - c)**2))


def dtw_distance(Q, C, return_path=False):
    """
    Classic Dynamic Time Warping distance.
    
    Args:
        Q: Query sequence (length n)
        C: Candidate sequence (length m)
        return_path: If True, also return warping path
    
    Returns:
        distance: DTW distance
        path (optional): Warping path as list of (i, j) tuples
    """
    n, m = len(Q), len(C)
    
    # Initialize cost matrix with infinity
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0  # Base case
    
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (Q[i-1] - C[j-1]) ** 2  # Local cost
            DTW[i, j] = cost + min(
                DTW[i-1, j],      # Insertion
                DTW[i, j-1],      # Deletion
                DTW[i-1, j-1]     # Match
            )
    
    distance = np.sqrt(DTW[n, m])
    
    if return_path:
        path = _backtrack(DTW, n, m)
        return distance, path
    
    return distance


def _backtrack(DTW, n, m):
    """
    Backtrack through cost matrix to find optimal warping path.
    
    Args:
        DTW: Filled cost matrix
        n, m: Dimensions
    
    Returns:
        path: List of (i, j) index pairs
    """
    i, j = n, m
    path = [(i, j)]
    
    while i > 0 and j > 0:
        # Find which neighbor had minimum cost
        candidates = [
            (i-1, j-1, DTW[i-1, j-1]),  # Diagonal
            (i-1, j, DTW[i-1, j]),      # Up
            (i, j-1, DTW[i, j-1])       # Left
        ]
        next_i, next_j, _ = min(candidates, key=lambda x: x[2])
        path.append((next_i, next_j))
        i, j = next_i, next_j
    
    # Handle boundary
    while i > 0:
        path.append((i-1, j))
        i -= 1
    while j > 0:
        path.append((i, j-1))
        j -= 1
    
    path.reverse()
    return path


def sakoe_chiba_dtw(Q, C, window_percent=0.10, return_path=False):
    """
    DTW with Sakoe-Chiba band constraint.
    
    Args:
        Q, C: Input sequences
        window_percent: Band width as fraction of sequence length
        return_path: If True, also return path
    
    Returns:
        distance: DTW distance with constraint
        path (optional): Warping path
    """
    n, m = len(Q), len(C)
    window = int(max(n, m) * window_percent)
    
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0
    
    for i in range(1, n + 1):
        # Apply window constraint
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
    
    if return_path:
        path = _backtrack(DTW, n, m)
        return distance, path
    
    return distance


# Example usage
if __name__ == "__main__":
    # Test with simple sequences
    np.random.seed(42)
    Q = np.sin(np.linspace(0, 2*np.pi, 50))
    C = np.sin(np.linspace(0, 2*np.pi, 50) * 1.1)  # Slightly stretched
    
    # Compute distances
    eucl = euclidean_distance(Q, C)
    dtw = dtw_distance(Q, C)
    sc_dtw = sakoe_chiba_dtw(Q, C, window_percent=0.15)
    
    print("Distance Comparison:")
    print(f"  Euclidean: {eucl:.4f}")
    print(f"  DTW:       {dtw:.4f}")
    print(f"  SC-DTW:    {sc_dtw:.4f}")
