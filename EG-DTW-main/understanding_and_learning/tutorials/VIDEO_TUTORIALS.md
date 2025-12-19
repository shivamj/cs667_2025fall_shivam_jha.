# Video Tutorials & Online Learning Resources

This file contains curated video tutorials, YouTube playlists, and online courses for learning DTW, entropy, and ECG signal processing. Videos are organized by difficulty and include timestamps for efficient learning.

---

## 🎬 Must-Watch Video Tutorials

### 1. Dynamic Time Warping Explained (StatQuest) ⭐⭐⭐⭐⭐
**Link**: https://www.youtube.com/watch?v=_K1OsqCicBY  
**Duration**: 11:26  
**Level**: Beginner  
**Best for**: First introduction to DTW concept

**What You'll Learn**:
- Visual intuition of DTW vs Euclidean distance
- How DTW handles time shifts
- Real-world applications

**Timestamps**:
- 0:00 - 2:30: Problem motivation (why Euclidean fails)
- 2:30 - 5:00: DTW concept visualization
- 5:00 - 8:00: Cost matrix explanation
- 8:00 - 11:26: Finding optimal path

**Action**: Watch this FIRST before reading any papers.

---

### 2. Dynamic Time Warping - MIT Lecture
**Search**: "MIT Dynamic Time Warping" OR "6.034 Dynamic Time Warping"  
**Link**: https://www.youtube.com/results?search_query=MIT+dynamic+time+warping  
**Duration**: ~20-30 minutes  
**Level**: Intermediate  
**Best for**: Understanding dynamic programming foundation

**What You'll Learn**:
- DP table filling strategy
- Recurrence relations
- Complexity analysis

**Note**: MIT OpenCourseWare lectures are excellent for rigorous understanding.

---

### 3. Time Series Classification with DTW (Python)
**Search**: "Python DTW tutorial" OR "Time series classification DTW python"  
**Recommended Channel**: Weights & Biases, ArjanCodes, or sentdex  
**Duration**: 15-30 minutes  
**Level**: Intermediate  
**Best for**: Hands-on implementation

**What You'll Learn**:
- DTW implementation in Python
- Using libraries (tslearn, dtaidistance)
- Classification with 1-NN

**Code-Along**: Follow along in Jupyter notebook

---

### 4. Shannon Entropy Explained (Khan Academy / 3Blue1Brown)
**Link**: https://www.youtube.com/results?search_query=shannon+entropy+explained  
**Duration**: 10-20 minutes  
**Level**: Beginner  
**Best for**: Understanding information theory basics

**What You'll Learn**:
- What is information?
- Entropy formula: H = -Σ p log p
- Intuition: Surprise and uncertainty

**Recommended Videos**:
- 3Blue1Brown: "Information Theory" (visual, excellent)
- Khan Academy: "Information Entropy" (step-by-step)

---

### 5. ECG Signal Processing Basics (PhysioNet)
**Search**: "ECG signal processing tutorial" OR "Pan Tompkins QRS detection"  
**Link**: https://www.youtube.com/results?search_query=ECG+signal+processing+basics  
**Duration**: 20-40 minutes  
**Level**: Intermediate  
**Best for**: Understanding ECG anatomy and preprocessing

**What You'll Learn**:
- P-QRS-T wave morphology
- Bandpass filtering (5-15 Hz)
- QRS detection algorithms
- Noise sources and removal

**Practical**: Look for videos showing Python/MATLAB implementations

---

### 6. Dynamic Programming Introduction (MIT / Abdul Bari)
**Link**: https://www.youtube.com/watch?v=oBt53YbR9Kk (Abdul Bari)  
**Duration**: 15-20 minutes  
**Level**: Beginner (prerequisite for DTW)  
**Best for**: DP fundamentals before DTW

**What You'll Learn**:
- Optimal substructure
- Overlapping subproblems
- Memoization vs tabulation
- Classic examples: Fibonacci, knapsack

**Why Important**: DTW is just 2D dynamic programming

---

## 📺 YouTube Playlists & Channels

### Recommended Channels

1. **StatQuest with Josh Starmer**
   - Link: https://www.youtube.com/@statquest
   - Focus: ML/stats concepts explained visually
   - DTW video: Excellent starter

2. **3Blue1Brown**
   - Link: https://www.youtube.com/@3blue1brown
   - Focus: Visual math explanations
   - Watch: Information theory series

3. **Arxiv Insights**
   - Link: https://www.youtube.com/@ArxivInsights
   - Focus: Research paper summaries
   - Good for: Understanding Soft-DTW and advanced topics

4. **sentdex (Python Programming)**
   - Link: https://www.youtube.com/@sentdex
   - Focus: Python tutorials
   - Search: Time series analysis playlists

5. **PhysioNet Channel**
   - Link: https://www.youtube.com/@PhysioNetOrg
   - Focus: Biomedical signal processing
   - Watch: ECG database tutorials

### Curated Playlists

**Time Series Analysis Playlist** (Various creators)
- Search: "Time series analysis python playlist"
- Duration: 2-5 hours total
- Topics: Preprocessing, DTW, classification, forecasting

**Dynamic Programming Playlist** (Abdul Bari / Back To Back SWE)
- Link: https://www.youtube.com/results?search_query=dynamic+programming+tutorial
- Duration: 4-6 hours
- Start with: Fibonacci, LCS, then DTW

**Signal Processing Fundamentals** (Steve Brunton)
- Channel: https://www.youtube.com/@Eigensteve
- Playlist: "Data-Driven Science and Engineering"
- Topics: Fourier, filtering, wavelets

---

## 🎓 Online Courses (Free & Paid)

### Free Courses

1. **MIT OpenCourseWare: Introduction to Algorithms**
   - Link: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/
   - Relevant: Lecture on Dynamic Programming
   - Time: 1-2 hours for DP section

2. **Stanford CS 229: Machine Learning**
   - Link: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
   - Relevant: Time series lecture
   - Time: ~1 hour

3. **Fast.ai: Practical Deep Learning**
   - Link: https://course.fast.ai/
   - Relevant: Time series classification module
   - Time: 2-3 hours

### Paid Courses (Worth It)

1. **Coursera: Time Series Forecasting**
   - Platform: Coursera
   - Instructor: Various
   - Search: "Time series analysis coursera"
   - Cost: $39-49/month (7-day free trial)
   - Completion: 4-6 weeks

2. **DataCamp: Time Series with Python**
   - Platform: DataCamp
   - Link: https://www.datacamp.com/
   - Cost: $25/month (first chapter free)
   - Completion: 4 hours

3. **Udemy: Time Series Analysis & Forecasting**
   - Platform: Udemy
   - Search: "Time series python udemy"
   - Cost: $10-20 (frequent sales)
   - Completion: 10-15 hours

**Note**: Free trials often sufficient for focused learning.

---

## 🔍 Search Queries for More Resources

Use these on YouTube, Google, or course platforms:

### Beginner Queries
```
"dynamic time warping explained"
"DTW tutorial for beginners"
"what is dynamic time warping"
"DTW visualization"
"dynamic programming tutorial"
```

### Intermediate Queries
```
"DTW python implementation"
"time series classification DTW"
"sakoe chiba band constraint"
"ECG signal processing python"
"shannon entropy tutorial"
```

### Advanced Queries
```
"soft DTW differentiable"
"fastDTW approximation"
"DTW lower bounding"
"multi-dimensional DTW"
"real-time DTW implementation"
```

---

## 📱 Mobile Learning Apps

### Recommended Apps

1. **Khan Academy** (iOS/Android)
   - Topics: Information theory, logarithms
   - Free, offline downloads

2. **Brilliant.org** (iOS/Android)
   - Topics: Dynamic programming, probability
   - Cost: $25/month (7-day free trial)

3. **YouTube** (iOS/Android)
   - Download videos for offline watching
   - Create playlists from links above

---

## 🎯 Learning Pathways by Video

### Fast Track (4 hours total)
1. StatQuest DTW (11 min)
2. Shannon Entropy (15 min)
3. Python DTW tutorial (30 min)
4. ECG basics (30 min)
5. Your own experiments (2+ hours)

### Standard Track (8-10 hours total)
1. Dynamic programming intro (30 min)
2. StatQuest DTW (11 min)
3. MIT DTW lecture (30 min)
4. Shannon Entropy (20 min)
5. ECG signal processing (40 min)
6. Python implementation tutorials (2 hours)
7. Your own experiments (4+ hours)

### Deep Dive (20+ hours total)
- All of Standard Track
- MIT 6.006 DP lectures (3 hours)
- Signal processing fundamentals (3 hours)
- Research paper walkthroughs (2 hours)
- Coursera/DataCamp course (10 hours)
- Advanced topics (Soft-DTW, FastDTW) (2+ hours)

---

## 📹 Video Notes Template

Use this template to take notes while watching:

```markdown
## Video: [Title]
**Link**: [URL]
**Date Watched**: [Date]
**Rating**: ⭐⭐⭐⭐⭐ (out of 5)

### Key Takeaways
1. 
2. 
3. 

### Timestamps & Notes
- 0:00 - [Topic]:  [Notes]
- 5:00 - [Topic]: [Notes]

### Questions to Research
- [ ] Question 1
- [ ] Question 2

### Action Items
- [ ] Try implementation shown in video
- [ ] Read paper mentioned
- [ ] Experiment with parameters

### Related Resources
- Paper: 
- Code: 
```

---

## 🎬 Video Watching Best Practices

### Active Watching Tips
1. **Don't just watch** - code along in real-time
2. **Pause frequently** - verify understanding at each step
3. **Take notes** - especially timestamps for key concepts
4. **Watch at 1.25-1.5x speed** - for review passes
5. **Watch twice** - first for overview, second for details

### Retention Strategies
1. **Feynman Technique**: Explain concept to someone after watching
2. **Spaced Repetition**: Re-watch videos after 1 day, 1 week, 1 month
3. **Implementation**: Code every algorithm shown within 24 hours
4. **Teaching**: Create your own mini-tutorial video

---

## 🔗 Supplementary Video Topics

### Math Foundations
- Logarithms review
- Probability distributions
- Linear algebra basics (for matrix operations)
- Calculus (derivatives - optional)

### Programming Foundations
- Python basics (if needed)
- NumPy tutorial
- Matplotlib plotting
- Jupyter notebooks

### Domain Knowledge
- ECG physiology
- Arrhythmia types
- Medical device regulations (if relevant)

---

## 📊 Video Learning Progress Tracker

Track what you've watched:

### Week 1: Foundations
- [ ] StatQuest DTW (11 min)
- [ ] Dynamic Programming intro (20 min)
- [ ] Shannon Entropy (15 min)
- [ ] Python basics review (30 min)

### Week 2: Implementation
- [ ] MIT DTW lecture (30 min)
- [ ] Python DTW tutorial (30 min)
- [ ] ECG processing (40 min)
- [ ] tslearn library tour (20 min)

### Week 3: Advanced
- [ ] Soft-DTW paper walkthrough (30 min)
- [ ] FastDTW explanation (20 min)
- [ ] Real-world applications (30 min)
- [ ] Your own mini-presentation (record yourself!)

---

## 🎤 Create Your Own Tutorial Video (Final Project)

After completing the course, record a 10-minute tutorial explaining:
1. What is DTW? (2 min)
2. How does it work? (3 min with example)
3. Why use entropy-adaptive constraints? (3 min)
4. Demo: Your implementation (2 min)

**Tools**:
- OBS Studio (free screen recording)
- Zoom (record yourself explaining)
- Loom (quick screen + cam)

**Benefits**:
- Solidifies your understanding
- Portfolio piece
- Teaching helps retention

---

## 💡 Video Learning Tips

### For Visual Learners
- Prioritize videos with animations and visualizations
- Pause to sketch diagrams yourself
- Use colored pens to annotate

### For Auditory Learners
- Listen to videos while commuting (audio-only mode)
- Repeat explanations out loud
- Discuss with study partners

### For Kinesthetic Learners
- Code along in real-time
- Build physical models (use objects to represent sequences)
- Walk through examples with hand gestures

---

**Last Updated**: November 24, 2025  
**Total Recommended Watch Time**: 4-20 hours (depending on track)  
**Best Starting Point**: StatQuest DTW video (11 min)

**Ready to start? Open the first video and let's learn! 🚀**
