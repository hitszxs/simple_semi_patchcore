# PatchCore EMD-Based Anomaly Detection (Custom Implementation)

## Introduction

Due to a recent server update, some files were lost. I’ve recovered the two most relevant Python files from my local machine, which are provided here.

This code is part of an improved version of the **PatchCore** method, mainly modifying the distance metric and the anomaly score computation logic.

---

## File Overview

- `emd.py`:  
  Contains two key functions:
  1. **EMD Distance Computation** — calculates the Earth Mover's Distance (EMD) between features.
  2. **Subsampling** — used to reduce memory bank size before distance computation.

- `pseduo.py`:  
  Handles memory bank sampling and inference. It performs the anomaly detection inference and produces pseudo labels based on the anomaly score.

---

## Important Note

If you are familiar with [anomalib](https://github.com/openvinotoolkit/anomalib), this should be straightforward.

This work is based on the `PatchCore` implementation within the anomalib project, with the following core changes:

- Replaced the distance metric with **EMD** (instead of cosine or Euclidean distance).
- Modified the **anomaly score computation** logic to match the EMD-based similarity behavior.

---