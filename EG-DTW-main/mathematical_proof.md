# Mathematical explanation and implementation details for EAC-DTW

This document gives a clear, step-by-step mathematical explanation of Dynamic Time Warping (DTW) and the Entropy-Adaptive Constraint DTW (EAC-DTW). It describes the matrices involved, the recurrence relations, an explicit numeric example, and a small Python-style implementation sketch. The level and wording are chosen so a 12th-grade student can follow.

**Goal:** Align two sequences (time series) Q and C while allowing non-linear stretching/compression, but prevent pathological alignments caused by noise using entropy-adaptive constraints.

**Notation**
- Let Q = (q_1, q_2, ..., q_n) be the query sequence of length n.
- Let C = (c_1, c_2, ..., c_m) be the candidate sequence of length m.
- Lowercase indices i (for Q) and j (for C) start at 1.

1) Basic pointwise cost

Define the local cost between two points as the squared difference:

$$
d(i,j) = (q_i - c_j)^2.
$$

We will compute a cost matrix D (of size (n+1)×(m+1)) that stores cumulative alignment costs. The extra row and column (index 0) are for boundary conditions.

2) DTW cumulative cost (recurrence)

Let D(i,j) be the minimum cumulative cost to align q_1..q_i with c_1..c_j. The classic DTW recurrence is:

$$
D(0,0)=0,\quad D(i,0)=D(0,j)=+\infty\text{ for }i>0\text{ or }j>0,
$$
and for i≥1, j≥1:

$$
D(i,j)=d(i,j)+\min\{D(i-1,j),\;D(i,j-1),\;D(i-1,j-1)\}.
$$

- Meaning: to reach cell (i,j) we come from one of three neighbors (up, left, diagonal) and add the local cost.
- The final DTW distance is sqrt(D(n,m)) (or simply D(n,m) if you keep squared distances).

3) Warping path

The warping path W is a sequence of index pairs W = ((i_1,j_1),(i_2,j_2),..., (i_L,j_L)) with (i_1,j_1)=(1,1) and (i_L,j_L)=(n,m). It follows the chosen transitions (diagonal, up, left). Backtracking from (n,m) using the choices that produced the minima gives the path.

4) Constraint bands (Sakoe–Chiba)

To prevent extreme warping, a simple constraint is the Sakoe–Chiba band: only allow j such that |i - j| ≤ w (constant w). This reduces computation and prevents pathological mapping where one point maps to many distant points.

5) Entropy-Adaptive Constraints (EAC-DTW)

EAC-DTW replaces the constant band width by a per-index window w_i that depends on local Shannon entropy of Q. Intuition:
- Low-entropy (flat/noisy) regions → tight window (small w_i)
- High-entropy (informative peaks like ECG QRS) → looser window (large w_i)

5.1) Local entropy computation

For each i compute entropy H_i using a small neighborhood (sliding window) around q_i:

1. Extract segment S_i = q_{i - r} ... q_{i + r} (window length B = 2r+1).
2. Compute histogram counts over B values into K bins, normalize to get probabilities p_k (k=1..K).
3. Shannon entropy (in bits):

$$
H_i = -\sum_{k=1}^K p_k \log_2(p_k + \varepsilon),
$$

where ε is a tiny positive number (e.g., 1e-10) to avoid log(0).

5.2) Sigmoid mapping to window sizes

Compute the mean entropy across the sequence μ_H = mean(H_i). Map H_i to window size w_i by a sigmoid function:

$$
w_i = w_{\min} + \frac{w_{\max} - w_{\min}}{1 + e^{-k (H_i - \mu_H)}}.
$$

Then convert to integer: W_i = floor(w_i). Choose k to control steepness. Typical choices: w_min=2, w_max = floor(α × n) with α≈0.10–0.20.

5.3) Use W_i as local constraint

For row i (q_i) allow j in [max(1, i - W_i), min(m, i + W_i)] only. Everywhere else treat D(i,j) = +∞ (skip computing).

6) Detecting singularities (fan-out)

Define a singularity as a run in the warping path where i stays the same while j increases (horizontal) or j stays the same while i increases (vertical). These indicate many-to-one mappings and are often caused by noise.

7) Worked numeric example (very small)

Let Q = [1, 2, 0, 1] (n=4) and C = [0, 1, 2, 1] (m=4).

Local costs d(i,j) = (q_i - c_j)^2 produce the following 4×4 matrix (rows i, columns j):

Compute each element:
- d(1,1)=(1-0)^2=1, d(1,2)=(1-1)^2=0, d(1,3)=(1-2)^2=1, d(1,4)=(1-1)^2=0
- d(2,1)=(2-0)^2=4, d(2,2)=1, d(2,3)=0, d(2,4)=1
- d(3,1)=(0-0)^2=0, d(3,2)=1, d(3,3)=4, d(3,4)=1
- d(4,1)=(1-0)^2=1, d(4,2)=0, d(4,3)=1, d(4,4)=0

So d =

$$
\begin{bmatrix}
1 & 0 & 1 & 0 \\
4 & 1 & 0 & 1 \\
0 & 1 & 4 & 1 \\
1 & 0 & 1 & 0
\end{bmatrix}
$$

Now fill cumulative matrix D with boundaries: use D(0,0)=0 and D(i,0)=D(0,j)=+∞.

Step-by-step (showing D(i,j)):
- D(1,1)=1 + min(inf,inf,0) = 1
- D(1,2)=0 + min(inf,1,inf) = 1
- D(1,3)=1 + min(inf,1,1) = 2
- D(1,4)=0 + min(inf,2,1) = 1

- D(2,1)=4 + min(1,inf,inf)=5
- D(2,2)=1 + min(1,5,1)=2
- D(2,3)=0 + min(2,2,1)=1
- D(2,4)=1 + min(1,1,2)=2

- D(3,1)=0 + min(5,inf,inf)=5
- D(3,2)=1 + min(2,5,5)=3
- D(3,3)=4 + min(1,3,2)=5
- D(3,4)=1 + min(2,5,1)=2

- D(4,1)=1 + min(5,inf,inf)=6
- D(4,2)=0 + min(3,6,5)=3
- D(4,3)=1 + min(5,3,3)=4
- D(4,4)=0 + min(2,4,5)=2

So D(4,4)=2 → sqrt(2) ≈ 1.414 is the DTW distance.

Backtracking from (4,4): choose predecessor with minimal D among (3,4),(4,3),(3,3).
- D(3,4)=2, D(4,3)=4, D(3,3)=5 → predecessor (3,4). Continue until (1,1) to extract path.

This small example shows how to compute the matrices by hand and confirms the recurrence.

8) Pseudocode / Python-style implementation sketch

```
def calculate_entropy(signal, window_size=11, bins=10):
	# sliding window entropy (return array H of length n)
	# use histogram probabilities and Shannon entropy

def sigmoid_window(H, w_min=2, w_max=30, k=2.0):
	mu = np.mean(H)
	w = w_min + (w_max - w_min) / (1 + np.exp(-k * (H - mu)))
	return np.floor(w).astype(int)

def eac_dtw(Q, C, w_min=2, w_max_percent=0.15, k=2.0):
	n, m = len(Q), len(C)
	w_max = int(max(n,m) * w_max_percent)
	H = calculate_entropy(Q)
	W = sigmoid_window(H, w_min, w_max, k)

	INF = 1e12
	D = np.full((n+1, m+1), INF)
	D[0,0] = 0

	for i in range(1, n+1):
		j_start = max(1, i - W[i-1])
		j_end   = min(m, i + W[i-1])
		for j in range(j_start, j_end+1):
			cost = (Q[i-1] - C[j-1])**2
			D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])

	# backtrack from (n,m) to get warping path
	return np.sqrt(D[n,m]), D
```

9) Complexity and numerical notes

- Classic DTW is O(n × m) time and O(n × m) memory.
- With local window W_i, complexity becomes roughly O(sum_i (2 W_i + 1)), often much smaller than nm.
- Entropy computation costs O(n × B) if B is the entropy window length (small constant).
- Use small epsilon in entropy and safe large INF for unreachable cells.

10) Practical tips

- Choose `w_min` small (e.g., 2) to limit fan-out in low-entropy regions.
- Choose `w_max_percent` around 0.10–0.20 depending on expected shifts.
- Choose `k` to control sensitivity (k higher → sharper transition between small and large windows).
- Use vectorized NumPy operations where possible for performance.

If you'd like, I can also:
- add a fully runnable Python file `eac_dtw.py` with unit tests, or
- generate a LaTeX-ready derivation for inclusion in your thesis.


To prove the validity of Entropy-Adaptive Constraint DTW (EAC-DTW) mathematically, we need to demonstrate how the constraint window 
w
i
w 
i
​
 
 directly limits the algorithm's ability to produce "Pathological Warping" (Singularities).
We will use a Proof by Constraints and a Matrix Iteration Example.
1. Mathematical Definitions
Let two time series be 
Q
=
{
q
1
,
…
,
q
n
}
Q={q 
1
​
 ,…,q 
n
​
 }
 and 
C
=
{
c
1
,
…
,
c
m
}
C={c 
1
​
 ,…,c 
m
​
 }
.
The Entropy Function
For a specific point 
q
i
q 
i
​
 
, we define the local entropy 
H
(
q
i
)
H(q 
i
​
 )
.
If region is flat (noise/isoelectric): 
H
(
q
i
)
→
0
H(q 
i
​
 )→0
.
If region is complex (QRS peak): 
H
(
q
i
)
→
H
m
a
x
H(q 
i
​
 )→H 
max
​
 
.
The Constraint Function (Sigmoid)
The window size 
w
i
w 
i
​
 
 at time step 
i
i
 is:
w
i
=
⌊
w
m
i
n
+
w
m
a
x
−
w
m
i
n
1
+
e
−
k
(
H
(
q
i
)
−
μ
)
⌋
w 
i
​
 =⌊w 
min
​
 + 
1+e 
−k(H(q 
i
​
 )−μ)
 
w 
max
​
 −w 
min
​
 
​
 ⌋
The Feasible Region
In Standard DTW, the feasible set of column indices 
j
j
 for row 
i
i
 is 
{
1
,
…
,
m
}
{1,…,m}
.
In EAC-DTW, the feasible set 
F
i
F 
i
​
 
 is strictly limited:
F
i
=
{
j
∣
i
−
w
i
≤
j
≤
i
+
w
i
}
F 
i
​
 ={j∣i−w 
i
​
 ≤j≤i+w 
i
​
 }
2. The Proof: Prevention of Pathological Warping
Objective: Prove that in a low-entropy (noisy) region, EAC-DTW prevents the alignment path from "stalling" (creating large horizontal or vertical segments) to match a noise spike.
Definition of a Singularity (Stalling):
A singularity occurs when a single point 
q
i
q 
i
​
 
 maps to a long sequence of points in 
C
C
: 
{
c
j
,
c
j
+
1
,
…
,
c
j
+
k
}
{c 
j
​
 ,c 
j+1
​
 ,…,c 
j+k
​
 }
.
Geometrically, this is a horizontal line of length 
k
k
 in the warping path.
Proof by Contradiction
Assumption:
Assume we are in a low-entropy region (flat line) where 
H
(
q
i
)
→
0
H(q 
i
​
 )→0
. Consequently, our mapping function gives the minimum window:
w
i
=
1
w 
i
​
 =1
Hypothesis:
Assume the algorithm tries to create a singularity (warp) of length 
k
=
3
k=3
 to dodge a noise spike.
The path would look like this: 
(
i
,
j
)
→
(
i
,
j
+
1
)
→
(
i
,
j
+
2
)
→
(
i
,
j
+
3
)
(i,j)→(i,j+1)→(i,j+2)→(i,j+3)
.
Verification against Constraints:
Let's check if these points exist in the Feasible Region 
F
i
F 
i
​
 
:
Point 
(
i
,
j
)
(i,j)
:
Constraint: 
∣
i
−
j
∣
≤
1
∣i−j∣≤1
. (Assume this is valid).
Point 
(
i
,
j
+
1
)
(i,j+1)
:
Constraint: 
∣
i
−
(
j
+
1
)
∣
∣i−(j+1)∣
. If 
j
=
i
j=i
, then 
∣
−
1
∣
=
1
∣−1∣=1
. (Valid).
Point 
(
i
,
j
+
2
)
(i,j+2)
:
Constraint: 
∣
i
−
(
j
+
2
)
∣
∣i−(j+2)∣
. If 
j
=
i
j=i
, then 
∣
−
2
∣
=
2
∣−2∣=2
.
Is 
2
≤
w
i
2≤w 
i
​
 
? No, because 
w
i
=
1
w 
i
​
 =1
.
Result: This cell is set to 
∞
∞
.
Conclusion:
In a low-entropy region where 
w
i
w 
i
​
 
 is small, it is mathematically impossible for the warping path to deviate significantly from the diagonal. The algorithm is forced to step diagonally 
(
i
+
1
,
j
+
1
)
(i+1,j+1)
, ignoring the "better" cost of the noise spike.
∴
∴
 Singularities are prevented.
3. Matrix Iteration Proof (Visual)
Let's trace a 
3
×
3
3×3
 matrix to see the values.
Scenario:
Query 
Q
Q
 (Clean): [0, 0, 0] (Flat line, Entropy 
≈
0
→
w
=
0
≈0→w=0
).
Candidate 
C
C
 (Noisy): [0, 10, 0] (Has a noise spike at index 1).
Constraint: Since 
Q
Q
 is flat, 
w
i
=
0
w 
i
​
 =0
 (Strict Diagonal).
Step 1: Initialize Matrix with Infinity
[
∞
∞
∞
∞
∞
∞
∞
∞
∞
]
​
  
∞
∞
∞
​
  
∞
∞
∞
​
  
∞
∞
∞
​
  
​
 
Step 2: Calculate Row 0 (
i
=
0
i=0
)
q
0
=
0
q 
0
​
 =0
. Entropy is low, so 
w
0
=
0
w 
0
​
 =0
.
Valid 
j
j
 range: 
0
±
0
→
{
0
}
0±0→{0}
.
Cost: 
(
q
0
−
c
0
)
2
=
(
0
−
0
)
2
=
0
(q 
0
​
 −c 
0
​
 ) 
2
 =(0−0) 
2
 =0
.
Accumulated: 
0
0
.
Matrix:
[
0
∞
∞
∞
∞
∞
∞
∞
∞
]
​
  
0
∞
∞
​
  
∞
∞
∞
​
  
∞
∞
∞
​
  
​
 
Step 3: Calculate Row 1 (
i
=
1
i=1
)
q
1
=
0
q 
1
​
 =0
. Entropy is low, so 
w
1
=
0
w 
1
​
 =0
.
Valid 
j
j
 range: 
1
±
0
→
{
1
}
1±0→{1}
. (Only index 1 is allowed!).
The Critical Moment:
Cell 
(
1
,
0
)
(1,0)
: INVALID (Outside constraint).
Cell 
(
1
,
1
)
(1,1)
: VALID.
Local Cost: 
(
q
1
−
c
1
)
2
=
(
0
−
10
)
2
=
100
(q 
1
​
 −c 
1
​
 ) 
2
 =(0−10) 
2
 =100
.
Neighbor: Only diagonal is available 
(
0
,
0
)
(0,0)
.
Total: 
100
+
0
=
100
100+0=100
.
Cell 
(
1
,
2
)
(1,2)
: INVALID.
Matrix:
[
0
∞
∞
∞
100
∞
∞
∞
∞
]
​
  
0
∞
∞
​
  
∞
100
∞
​
  
∞
∞
∞
​
  
​
 
Note: Standard DTW might have tried to warp to a 0 to avoid the 100. EAC-DTW forces us to take the hit.
Step 4: Calculate Row 2 (
i
=
2
i=2
)
q
2
=
0
q 
2
​
 =0
. Entropy is low, so 
w
2
=
0
w 
2
​
 =0
.
Valid 
j
j
 range: 
2
±
0
→
{
2
}
2±0→{2}
.
Cell 
(
2
,
2
)
(2,2)
 Cost: 
(
0
−
0
)
2
=
0
(0−0) 
2
 =0
.
Neighbor: 
(
1
,
1
)
=
100
(1,1)=100
.
Total: 
100
100
.
Final Matrix:
[
0
∞
∞
∞
100
∞
∞
∞
100
]
​
  
0
∞
∞
​
  
∞
100
∞
​
  
∞
∞
100
​
  
​
 
Comparison Interpretation
EAC-DTW Path: 
(
0
,
0
)
→
(
1
,
1
)
→
(
2
,
2
)
(0,0)→(1,1)→(2,2)
. It traveled the Diagonal.
It acknowledged the noise (Cost 100) but kept the timing correct.
Standard DTW Path: Would likely go 
(
0
,
0
)
→
(
0
,
1
)
→
(
0
,
2
)
(0,0)→(0,1)→(0,2)
 (Singularity).
It would try to match 
Q
0
(
0
)
Q 
0
​
 (0)
 with 
C
2
(
0
)
C 
2
​
 (0)
 to get 0 cost.
Result: The time axis is broken. The signal is distorted.
Summary Equation for Thesis
You can summarize this proof with this single conditional equation:
D
(
i
,
j
)
=
{
(
q
i
−
c
j
)
2
+
min
⁡
(
…
)
if 
∣
i
−
j
∣
≤
w
(
H
(
q
i
)
)
∞
otherwise
D(i,j)={ 
(q 
i
​
 −c 
j
​
 ) 
2
 +min(…)
∞
​
  
if ∣i−j∣≤w(H(q 
i
​
 ))
otherwise
​
 
As 
H
(
q
i
)
→
0
H(q 
i
​
 )→0
, then 
w
(
H
)
→
0
w(H)→0
, forcing 
∣
i
−
j
∣
→
0
∣i−j∣→0
 (Diagonal Identity).