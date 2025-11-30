# Grand X-Ray Slam – Division A & B  
## 2nd Place (Division A) & 3rd Place (Division B)

This repository contains my complete end-to-end solution for the **Grand X-Ray Slam Kaggle Competition (Division A & B)**.  
I implemented a two-model ensemble using **EVA-X** and **CheXFound**, combined with optimized preprocessing, strong augmentation strategies, and checkpoint averaging to achieve top leaderboard results.

---

# Competition
**Grand X-Ray Slam (Division A & B)**  
Chest X-ray Abnormality Classification  
Kaggle Page: https://www.kaggle.com/competitions/grand-xray-slam-division-a

---

# Overview of My Approach

This solution includes:

- Preprocessing pipeline with **600×600** image resizing  
- Two high-performance vision models:
  - **EVA-X (448×448)**
  - **CheXFound (512×512)**
- Layer-wise learning rates  
- Mixed Precision (AMP)  
- Focal Loss + Binary Cross-Entropy Loss  
- Exponential Moving Average (EMA)  
- Strong geometric, photometric, and dropout-based augmentations  
- Checkpoint averaging (epochs 5 & 6)  
- **Two-model ensemble** for final predictions  
- Optimized inference (reduced runtime from ~9 hours to ~7 hours)

All Division A and Division B models were trained **exclusively on their respective datasets**.

---

# Preprocessing

All X-ray images were downsampled to **600×600** to reduce I/O overhead and improve data loading efficiency.

### Notebooks
- **Division A:** `600-p-div-a.ipynb`  
- **Division B:** `600-p-div-b.ipynb`

---

# Models

## 1. EVA-X Base Model  
Repository: EVA-X (GitHub)  
Image Size: **448×448**  
Training Duration: **6 epochs**

### Training Strategy
- Layer-wise learning rates  
- Focal Loss + BCE  
- Strong augmentations (geometric, photometric, dropout-based)  
- Mixed Precision (AMP)  
- Exponential Moving Average (EMA)  
- Checkpoint averaging (epochs 5 + 6)

### Notebooks
- **Division A:** `evax-recs-448-div-a.ipynb`  
- **Division B:** `evax-recs-448-div-b.ipynb`

---

## 2. CheXFound Model  
Repository: CheXFound (GitHub)  
Image Size: **512×512**  
Training Duration: **6 epochs (3 + 3)**

### Training Strategy
- Only the **GLoRIA head** was fine-tuned  
- Backbone remained frozen  
- Same augmentations & optimizations as EVA-X  

### Notebooks
- **Div A (epochs 1–3):** `all-data-chexfound-recs-div-a_3.ipynb`  
- **Div A (epochs 4–6):** `all-data-chexfound-recs-div-a_6.ipynb`  
- **Div B (epochs 1–3):** `all-data-chexfound-recs-div-b_3.ipynb`  
- **Div B (epochs 4–6):** `all-data-chexfound-recs-div-b_6.ipynb`

---

# Inference & Ensembling

Final predictions used both models:

### EVA-X
- Epochs **5 and 6**  
- With **Test-Time Augmentation (TTA)**  

### CheXFound
- Epochs **5 and 6**  
- Without TTA

### Final Ensemble
The final submission was the **average of**:
- EVA-X (epoch 5 + epoch 6 + TTA)  
- CheXFound (epoch 5 + epoch 6)

### Inference Notebooks
- **Division A:** `all-models-inference-div-a.ipynb`  
- **Division B:** `all-models-inference-div-b.ipynb`

---

# Competition Results

| Division     | Rank       | Notes                                      |
|--------------|------------|---------------------------------------------|
| Division A   | **2nd Place** | EVA-X + CheXFound ensemble                 |
| Division B   | **3rd Place** | Same pipeline, trained on Div B only       |

---

# Acknowledgements
- EVA-X authors  
- CheXFound team  
- Kaggle competition organizers  
