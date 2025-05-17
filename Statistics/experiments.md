# Experimental Results Summary

1. **Statistical Approach + Magi Detection Annotations**: Feature engineering using annotations extracted with Magi, followed by traditional ML classification.
2. **Zero-Shot Approach**: Direct classification using pre-trained vision-language models (CLIP and SigLIP).

## Summary Table

| Approach                            | Task        | Accuracy | F1 Macro avg     |
|-------------------------------------|-------------|----------|----------------|
| Statistical + Magi (Random Forest)  | Binary      | 96.38%   | **0.92**     |
| Statistical + Magi (XGBoost)        | Multiclass  | 94.98%   | **0.80**     |
| CLIP (Zero-Shot)                    | Multiclass  | 92.80%   | **0.74**     |
| SigLIP (Zero-Shot)                  | Multiclass  | 89.41%   | **0.76**     |


---

## 1. Statistical Approach + Detection Annotations

**Binary Classification**  

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.81      | 0.91   | 0.86     | 120     |
| 1     | 0.98      | 0.97   | 0.98     | 737     |

**Multiclass Classification**  

| Class         | Precision | Recall | F1-Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Advertisement | 0.79      | 0.71   | 0.74     | 68      |
| Cover         | 0.82      | 0.70   | 0.76     | 20      |
| Story         | 0.97      | 0.99   | 0.98     | 737     |
| Textstory     | 0.74      | 0.72   | 0.73     | 32      |

---

## 2. Zero-Shot Approach

### CLIP Zero-Shot

**Per-Class Results**:
| Class         | Accuracy | Correct / Total |
|---------------|----------|-----------------|
| Advertisement | 0.9167   | 11 / 12         |
| Cover         | 0.7143   | 5 / 7           |
| Story         | 0.9563   | 197 / 206       |
| Textstory     | 0.5455   | 6 / 11          |

---

### SigLIP Zero-Shot

**Per-Class Results**:
| Class         | Accuracy | Correct / Total |
|---------------|----------|-----------------|
| Advertisement | 0.9167   | 11 / 12         |
| Cover         | 0.8571   | 6 / 7           |
| Story         | 0.9029   | 186 / 206       |
| Textstory     | 0.7273   | 8 / 11          |

---


## Linear Probing

### CLIP Linear Probing

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Advertisement    | 0.75      | 0.68   | 0.71     | 68      |
| Cover            | 0.33      | 0.88   | 0.48     | 16      |
| Story            | 0.98      | 0.96   | 0.97     | 739     |
| Textstory        | 0.80      | 0.75   | 0.77     | 32      |
| **Accuracy**     |           |        | **0.93** | 855     |
| **Macro Avg**    | 0.72      | 0.81   | 0.73     | 855     |
| **Weighted Avg** | 0.94      | 0.93   | 0.93     | 855     |
