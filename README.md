# 🔍 Symmetric Non-negative Matrix Factorization (SymNMF) Project

Welcome to our clustering adventure! This repo contains a full implementation of **Symmetric Non-negative Matrix Factorization (SymNMF)** — a graph-based clustering algorithm — built from scratch in **C and Python**, with a clean Python C API interface. We also benchmark it against **K-means** using silhouette scores to see how it stacks up.

> Built with determination and caffeine by [Adan-Assi](https://github.com/Adan-Assi) and [shaimahoji](https://github.com/shaimahoji)

---

## 📦 What's Inside

- 🐍 `symnmf.py` — Python interface for running SymNMF and matrix operations  
- ⚙️ `symnmf.c/.h` — C implementation of core matrix logic  
- 🔗 `symnmfmodule.c` — Python C API wrapper to connect Python with C  
- 📊 `analysis.py` — Compare SymNMF vs K-means using `sklearn.metrics.silhouette_score`  
- 🛠️ `setup.py` — Build script for the Python extension  
- 🧱 `Makefile` — Build script for the C executable  
- 🧩 Optional `.c/.h` modules — for modularity and clean design  
---

## ⚙️ How It Works

SymNMF clusters data by factorizing a normalized similarity matrix \( W \) into a low-rank non-negative matrix \( H \), such that:

**Objective:** Minimize ‖W − H·Hᵗ‖² subject to H ≥ 0


Each row in \( H \) gives a soft association score to clusters. We convert this to hard clustering by picking the max value per row.

We also implemented:
- Similarity matrix computation using Gaussian kernel  
- Degree matrix and normalized Laplacian  
- Iterative update rule for optimizing \( H \)  
- Convergence check with tolerance \( $\epsilon$ = 1e-4 \) and max_iter = 300  

---

## 🧪 How to Run

### 🧠 Run SymNMF from Python
```bash
python3 symnmf.py 3 symnmf input.txt
```

### ⚙️ Run matrix operations from C
```bash
make
./symnmf norm input.txt
```

### 📊 Compare SymNMF vs K-means
Example command line:
```bash
python3 analysis.py 3 input.txt
```

---

## 📈 Sample Output
```text
>>> python3 symnmf.py 2 symnmf input_1.txt
0.0600,0.0100
0.0100,0.0500
0.0100,0.0400
0.0200,0.0400
0.0500,0.0200
```

```text
>>> python3 analysis.py 5 input_k5_d7.txt
nmf: 0.1162
kmeans: 0.1147
```

---

## 🧠 Why This Project Matters
- 🛠️ Implements clustering from scratch — no external ML libraries required  
- 📐 Combines numerical optimization, graph theory, and matrix algebra in a clean pipeline  
- 🔌 Bridges Python and C for performance-critical tasks using a custom C extension  
- 🚀 Offers hands-on experience building and benchmarking algorithms with real metrics  

---

## 🧼 Notes & Assumptions
- 🔢 All outputs are formatted to **4 decimal places** for consistency  
- ⚠️ Error handling is minimal: prints `"An Error Has Occurred"` and exits gracefully  
- 🧹 Manual memory management in C — yes, we cleaned up after ourselves  
- 📄 Input files are plain `.txt` with one data point per line  
- 🧳 No containers or frameworks used — designed to run directly on **Nova**  

---

## 📚 Reference
Da Kuang, Chris Ding, and Haesun Park. *Symmetric nonnegative matrix factorization for graph clustering.* SIAM SDM, 2012.

---

Thanks for checking out our work!  

