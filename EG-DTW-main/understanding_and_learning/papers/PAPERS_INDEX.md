# Papers Index & Resource Links

This file contains direct links to all essential papers, preprints, and downloadable resources for learning DTW and EAC-DTW.

---

## 📄 Core DTW Papers

### 1. Sakoe & Chiba (1978) - The Original DTW Paper
**Title**: "Dynamic programming algorithm optimization for spoken word recognition"  
**Citation**: IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43-49.

**Download Links**:
- IEEE Xplore: https://ieeexplore.ieee.org/document/1163055
- Alternative (requires university access): Search on Google Scholar
- Key Innovation: Sakoe-Chiba band constraint

**Summary**:
This is the foundational paper that introduced constrained DTW for speech recognition. They proposed the "Sakoe-Chiba band" - a diagonal corridor of fixed width around the main diagonal of the cost matrix. This reduces computation from O(n²) to O(n·w) where w is the band width.

**Key Sections to Read**:
- Section II: DP matching algorithm
- Section III: Local path constraints
- Figure 3: Illustration of band constraint

**One-Sentence Takeaway**: DTW with a fixed diagonal band prevents pathological warping and reduces computation.

---

### 2. Ratanamahatana & Keogh (2004) - "Everything you know about DTW is wrong"
**Title**: "Everything you know about Dynamic Time Warping is Wrong"  
**Citation**: 3rd Workshop on Mining Temporal and Sequential Data (2004)

**Download Links**:
- Direct PDF: https://www.cs.ucr.edu/~eamonn/DTW_myths.pdf
- Author's website: https://www.cs.ucr.edu/~eamonn/
- Google Scholar: Search "Everything you know about DTW is wrong"

**Summary**:
This paper debunks common misconceptions about DTW:
- Myth 1: DTW doesn't work on long sequences → FALSE (with constraints)
- Myth 2: You need to normalize data → TRUE (but with caveats)
- Myth 3: Euclidean is faster and as good → FALSE
- Myth 4: Lower bounding speeds up search → TRUE (with UCR suite)

**Key Sections**:
- Section 3: Seven myths about DTW
- Section 5: Empirical validation

**One-Sentence Takeaway**: Most "common knowledge" about DTW limitations is outdated or wrong; proper implementation makes it highly effective.

---

### 3. Müller (2007) - Information Retrieval for Music and Motion
**Title**: "Information Retrieval for Music and Motion"  
**Publisher**: Springer  
**Chapter**: 4 (Dynamic Time Warping)

**Download Links**:
- Springer: https://www.springer.com/gp/book/9783540740476
- University library access required
- Alternative: Search for "Meinard Müller DTW tutorial" for lecture slides

**Summary**:
Comprehensive tutorial on DTW variants including:
- Step patterns (symmetric vs asymmetric)
- Global constraints (Sakoe-Chiba, Itakura)
- Normalization strategies
- Multi-dimensional DTW

**Key Sections**:
- Chapter 4.2: Classic DTW
- Chapter 4.3: DTW variants
- Chapter 4.4: Acceleration techniques

**One-Sentence Takeaway**: Authoritative reference covering all DTW flavors and best practices for different domains.

---

### 4. Cuturi & Blondel (2017) - Soft-DTW
**Title**: "Soft-DTW: a differentiable loss function for time-series"  
**Citation**: ICML 2017

**Download Links**:
- arXiv: https://arxiv.org/abs/1703.01541
- PDF: https://arxiv.org/pdf/1703.01541.pdf
- Code: https://github.com/mblondel/soft-dtw

**Summary**:
Instead of hard min{·,·,·} in the DTW recurrence, Soft-DTW uses a smooth approximation:
```
softmin(a,b,c) = -γ log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))
```
This makes DTW differentiable, enabling gradient-based learning. Note: This is a different approach than EAC-DTW (smoothing vs adaptive constraints).

**Key Sections**:
- Section 2: Soft-DTW definition
- Section 3: Gradient computation
- Section 5: Experiments

**One-Sentence Takeaway**: Soft-DTW enables end-to-end gradient descent training but doesn't address pathological warping like EAC-DTW does.

---

## 🔬 ECG & Signal Processing Papers

### 5. Pan & Tompkins (1985) - QRS Detection
**Title**: "A Real-Time QRS Detection Algorithm"  
**Citation**: IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236

**Download Links**:
- IEEE Xplore: https://ieeexplore.ieee.org/document/4122029
- PubMed: https://pubmed.ncbi.nlm.nih.gov/3997178/
- Alternative: Search "Pan Tompkins QRS detection PDF"

**Summary**:
The gold standard for ECG QRS complex detection. Describes:
- Bandpass filter: 5-15 Hz (removes baseline wander and high-frequency noise)
- Derivative filter: Highlights QRS slope
- Squaring: Emphasizes large derivatives
- Moving window integration: Smooths detection
- Adaptive thresholding: Finds peaks

**Key Sections**:
- Section II: Preprocessing (filtering)
- Section III: QRS detection algorithm
- Figure 1: Filter response

**One-Sentence Takeaway**: Standard preprocessing pipeline (bandpass + derivative) preserves QRS complex while removing noise.

---

### 6. Moody & Mark (2001) - MIT-BIH Database
**Title**: "The impact of the MIT-BIH Arrhythmia Database"  
**Citation**: IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50

**Download Links**:
- IEEE: https://ieeexplore.ieee.org/document/932724
- PhysioNet: https://physionet.org/content/mitdb/1.0/
- Database access: https://archive.physionet.org/physiobank/database/mitdb/

**Summary**:
Describes the creation and impact of the MIT-BIH Arrhythmia Database:
- 48 ECG recordings (30 min each, 360 Hz)
- ~110,000 annotated beats
- 5 main arrhythmia classes + 15 subclasses
- Gold standard for algorithm validation

**Key Information**:
- Access: Free via PhysioNet (requires registration)
- Format: WFDB format (use `wfdb` Python library)
- Annotations: Expert-labeled beat types

**One-Sentence Takeaway**: MIT-BIH is the standard benchmark dataset for ECG classification research.

---

## 📊 Information Theory Papers

### 7. Shannon (1948) - Entropy
**Title**: "A Mathematical Theory of Communication"  
**Citation**: Bell System Technical Journal, 27(3), 379-423

**Download Links**:
- Original: http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
- Reprinted: https://www.jstor.org/stable/3611062
- Modern tutorial: Search "Shannon entropy tutorial"

**Summary**:
The birth of information theory. Defines entropy as:
```
H(X) = -Σ P(x) log₂ P(x)
```
Interprets entropy as:
- Average information content
- Minimum bits needed to encode a message
- Measure of uncertainty/surprise

**Key Sections**:
- Part I: Discrete noiseless systems (pages 1-11)
- Definition of entropy (page 10)
- Properties of entropy (page 11)

**One-Sentence Takeaway**: Entropy quantifies the "information content" or "complexity" of a signal - foundational for EAC-DTW's adaptive constraints.

---

## 🎓 Tutorial Papers & Reviews

### 8. Senin (2008) - DTW for Data Mining
**Title**: "Dynamic Time Warping Algorithm Review"  
**Citation**: Information and Computer Science Department, University of Hawaii

**Download Links**:
- Direct PDF: http://seninp.github.io/assets/pubs/senin_dtw_litreview_2008.pdf
- Author's page: http://seninp.github.io/

**Summary**:
Excellent literature review covering:
- DTW history and applications
- Algorithmic variants
- Acceleration techniques
- Common pitfalls

**One-Sentence Takeaway**: Gentle introduction to DTW with clear examples and comprehensive references.

---

### 9. Keogh & Ratanamahatana (2005) - Exact Indexing
**Title**: "Exact indexing of dynamic time warping"  
**Citation**: Knowledge and Information Systems, 7(3), 358-386

**Download Links**:
- Springer: https://link.springer.com/article/10.1007/s10115-004-0154-9
- UCR Eamonn's page: https://www.cs.ucr.edu/~eamonn/

**Summary**:
Describes lower bounding techniques to speed up DTW search:
- LB_Keogh: Lower bound using envelope
- Early abandoning: Stop computation if partial distance exceeds threshold
- Achieves 2-3 orders of magnitude speedup

**One-Sentence Takeaway**: DTW can be as fast as Euclidean distance with proper indexing and pruning.

---

## 📚 Books (Chapter References)

### 10. Bishop (2006) - Pattern Recognition and Machine Learning
**Chapter**: 14.4 (Hidden Markov Models - includes sequence alignment)

**Download Links**:
- Publisher: https://www.springer.com/gp/book/9780387310732
- PDF (check university library or author's website)

**Relevant Sections**:
- HMM forward algorithm (analogous to DTW)
- Viterbi algorithm (path extraction)

---

### 11. Hastie et al. (2009) - Elements of Statistical Learning
**Chapter**: 14.7 (Hidden Markov Models)

**Download Links**:
- Free PDF: https://web.stanford.edu/~hastie/ElemStatLearn/
- Publisher: Springer

**Relevant Sections**:
- Dynamic programming for sequence analysis
- Viterbi algorithm

---

## 🌐 Online Resources

### Interactive Tutorials
1. **Databricks DTW Blog**
   - URL: https://databricks.com/blog/2019/04/30/understanding-dynamic-time-warping.html
   - Interactive visualizations

2. **Alexminnaar's DTW Tutorial**
   - URL: http://alexminnaar.com/2014/04/16/time-series-classification-and-clustering-with-python.html
   - Python code examples

3. **tslearn Documentation**
   - URL: https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
   - Official library docs with examples

### GitHub Repositories
1. **dtaidistance**
   - URL: https://github.com/wannesm/dtaidistance
   - Fast DTW implementations in C

2. **tslearn**
   - URL: https://github.com/tslearn-team/tslearn
   - Machine learning toolkit for time series

3. **UCR Suite**
   - URL: https://www.cs.ucr.edu/~eamonn/UCRsuite.html
   - Optimized DTW for large datasets

### Datasets
1. **UCR Time Series Archive**
   - URL: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
   - 128 time series datasets for benchmarking

2. **PhysioNet Databases**
   - URL: https://physionet.org/about/database/
   - Medical time series (ECG, EEG, etc.)

3. **MIT-BIH Arrhythmia Database**
   - URL: https://physionet.org/content/mitdb/1.0/
   - Direct access to ECG data

---

## 📥 How to Download Papers

### Method 1: Direct Links
Click the links above. If you hit a paywall, try Method 2 or 3.

### Method 2: University Library Access
```powershell
# If you have university credentials:
# 1. Go to your university library website
# 2. Search for the paper title
# 3. Download PDF through institutional access
```

### Method 3: Author's Websites
Many authors post PDFs on their personal pages:
- Eamonn Keogh: https://www.cs.ucr.edu/~eamonn/
- Meinard Müller: https://www.audiolabs-erlangen.de/fau/professor/mueller
- Marco Cuturi: https://marcocuturi.net/

### Method 4: arXiv / Preprint Servers
Search on:
- arXiv: https://arxiv.org/
- bioRxiv (for medical papers): https://www.biorxiv.org/

### Method 5: Google Scholar
```
1. Go to https://scholar.google.com/
2. Search paper title
3. Look for [PDF] link on the right
4. Often free versions available
```

### Method 6: Request from Authors
Email authors directly - most are happy to share their work for educational purposes.

---

## 📖 Recommended Reading Order

### For Beginners (Start Here)
1. Senin (2008) - DTW review
2. Ratanamahatana & Keogh (2004) - DTW myths
3. Shannon entropy tutorial (online)
4. Sakoe & Chiba (1978) - skim algorithm section

### For Intermediate Learners
5. Müller (2007) - Chapter 4
6. Pan & Tompkins (1985) - ECG preprocessing
7. Shannon (1948) - Part I (entropy)

### For Advanced Study
8. Cuturi (2017) - Soft-DTW
9. Keogh & Ratanamahatana (2005) - Indexing
10. Your own thesis chapter on EAC-DTW!

---

## ✅ Download Checklist

Track your progress:

- [ ] Sakoe & Chiba (1978) PDF saved
- [ ] Ratanamahatana & Keogh (2004) PDF saved
- [ ] Müller (2007) Chapter 4 accessed
- [ ] Cuturi (2017) arXiv PDF saved
- [ ] Pan & Tompkins (1985) PDF saved
- [ ] Shannon (1948) PDF saved
- [ ] Senin (2008) review saved
- [ ] Watched at least 2 video tutorials
- [ ] Bookmarked tslearn documentation
- [ ] Registered on PhysioNet (for MIT-BIH access)

---

## 🔗 Quick Links Summary

| Resource | Type | Link |
|----------|------|------|
| Sakoe & Chiba (1978) | Paper | https://ieeexplore.ieee.org/document/1163055 |
| Ratanamahatana DTW Myths | Paper | https://www.cs.ucr.edu/~eamonn/DTW_myths.pdf |
| Cuturi Soft-DTW | arXiv | https://arxiv.org/abs/1703.01541 |
| Pan & Tompkins QRS | Paper | https://pubmed.ncbi.nlm.nih.gov/3997178/ |
| Shannon Entropy | Paper | http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf |
| tslearn Docs | Tutorial | https://tslearn.readthedocs.io/ |
| MIT-BIH Database | Dataset | https://physionet.org/content/mitdb/1.0/ |
| UCR Archive | Dataset | https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ |
| Databricks DTW Blog | Tutorial | https://databricks.com/blog/2019/04/30/understanding-dynamic-time-warping.html |

---

**Last Updated**: November 24, 2025  
**Maintainer**: Fnu Ashutosh
