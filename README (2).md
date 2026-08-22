# 🫁 PneumoVision AI

### Chest X-ray Pneumonia Classification with EfficientNetB0 + Grad-CAM

PneumoVision AI is an educational computer-vision project that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** using a fine-tuned **EfficientNetB0** model. It also uses **Grad-CAM** to visualize regions that contributed strongly to the model's prediction.

> **Medical disclaimer:** This project is for education and demonstration only. It is not clinically validated and must not be used to diagnose disease, make treatment decisions, or replace a qualified healthcare professional.

---

## 1. Project Overview

The trained pipeline is:

```text
Chest X-ray
    ↓
RGB conversion
    ↓
224 × 224 resize
    ↓
Data augmentation layer
    ↓
EfficientNetB0
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.4)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(1, Sigmoid)
    ↓
P(PNEUMONIA)
```

The notebook used transfer learning followed by fine-tuning, class weights for class imbalance, and evaluation with accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC and Grad-CAM.

---

## 2. Dataset

The training notebook used the Kaggle **Chest X-Ray Images (Pneumonia)** dataset with two classes:

- `NORMAL`
- `PNEUMONIA`

Reported training counts:

| Class | Images |
|---|---:|
| NORMAL | 1,342 |
| PNEUMONIA | 3,876 |

Reported untouched test set:

| Class | Images |
|---|---:|
| NORMAL | 234 |
| PNEUMONIA | 390 |
| **Total** | **624** |

The original validation folder was very small, so the notebook created an 80/20 validation split from the training directory using a fixed seed.

The deployed Streamlit app **does not require the Kaggle dataset**. It only needs the saved model and an uploaded X-ray.

---

## 3. Model Architecture

The saved model is:

```text
pneumonia_efficientnetb0.keras
```

The trained model contains the following outer layers:

```text
input_layer
↓
data_augmentation
↓
efficientnetb0
↓
global_average_pooling2d
↓
dropout
↓
dense
↓
dropout_1
↓
dense_1
```

The EfficientNetB0 base was created with `include_top=False`, initialized from ImageNet weights, and fine-tuned in the upper portion of the network.

### Data augmentation used during training

```text
RandomFlip("horizontal")
RandomRotation(0.05)
RandomZoom(0.10)
RandomContrast(0.10)
```

### Classification head

```text
GlobalAveragePooling2D
Dropout(0.4)
Dense(128, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")
```

The sigmoid output is treated as `P(PNEUMONIA)` because the directory-based training labels are ordered as `NORMAL = 0` and `PNEUMONIA = 1`.

The Streamlit application uses a 0.50 threshold:

```text
P(PNEUMONIA) >= 0.50 → PNEUMONIA
P(PNEUMONIA) <  0.50 → NORMAL
```

---

## 4. Input Processing

The deployed application follows the notebook's input format:

1. Load the uploaded image.
2. Convert it to RGB.
3. Resize it to `224 × 224`.
4. Convert to `float32`.
5. Add a batch dimension.
6. Pass it to the saved Keras model.

The deployment app does **not** manually divide the image by 255 because the training pipeline did not add a separate `Rescaling(1./255)` layer. EfficientNet's Keras preprocessing behavior remains part of the model.

---

## 5. Class Imbalance

The training data contained substantially more pneumonia images than normal images. The notebook addressed this using balanced class weights with `compute_class_weight` and supplied those weights to model training.

This helps reduce the model's tendency to favor the majority class.

---

## 6. Training and Fine-Tuning

### Stage 1 — Transfer learning

EfficientNetB0 was initialized using ImageNet weights while its convolutional base was frozen. A new classification head was trained using:

- Adam optimizer
- Learning rate: `1e-4`
- Binary cross-entropy loss
- Accuracy
- Precision
- Recall
- ROC-AUC
- Early stopping
- ReduceLROnPlateau
- Model checkpointing
- Class weights

### Stage 2 — Fine-tuning

The upper portion of EfficientNetB0 was unfrozen while approximately the first 80% of the base layers remained frozen.

Fine-tuning used:

- Adam optimizer
- Learning rate: `1e-5`
- Binary cross-entropy loss
- Validation-loss model checkpointing
- Early stopping
- ReduceLROnPlateau

---

## 7. Final Test Performance

The completed test evaluation reported:

| Metric | Result |
|---|---:|
| Test images | 624 |
| Accuracy | **86.38%** |
| ROC-AUC | **0.95035** |
| Pneumonia recall | **97%** |

Classification report:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| NORMAL | 0.93 | 0.69 | 0.79 |
| PNEUMONIA | 0.84 | 0.97 | 0.90 |

Confusion matrix:

```text
                  Predicted
                NORMAL  PNEUMONIA

Actual NORMAL      161       73
Actual PNEUMONIA    12      378
```

These are experimental test-set results from the completed notebook and are not clinical performance guarantees.

---

## 8. Grad-CAM Explainability

Grad-CAM is generated from the nested EfficientNetB0 model's `top_conv` layer, matching the working Grad-CAM implementation used in the notebook.

The app displays two views:

1. **Attention Heatmap** — a color overlay showing model activation strength.
2. **AI-Highlighted Region** — a bounding rectangle around the strongest sufficiently localized activation.

The highlighted region is intentionally described as an **AI-highlighted attention region**, not a confirmed lesion or medically segmented defect.

Grad-CAM can be diffuse or misleading and should not be treated as a medical localization method.

---

## 9. Streamlit Features

The application provides:

- Chest X-ray upload
- Original image preview
- One-click analysis button
- NORMAL/PNEUMONIA prediction
- Confidence score
- NORMAL probability
- PNEUMONIA probability
- Probability bars
- Grad-CAM heatmap
- AI-highlighted region
- Approximate activation region information
- Model information sidebar
- Reported test metrics
- Medical disclaimer
- Responsive wide layout

---

## 10. Repository Structure

The GitHub repository should contain exactly these deployment files:

```text
PneumoVision-AI/
├── app.py
├── pneumonia_efficientnetb0.keras
├── requirements.txt
└── README.md
```

Do **not** upload the full Kaggle dataset. It is not required for inference.

---

## 11. Requirements

Recommended `requirements.txt`:

```text
streamlit>=1.48,<2
tensorflow==2.20.0
opencv-python-headless
numpy
pillow
```

TensorFlow is pinned to the training environment version used by the project.

---

## 12. Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

The model file must be beside `app.py`:

```text
app.py
pneumonia_efficientnetb0.keras
```

The app loads the model using a path relative to `app.py`, so it does not depend on Kaggle paths such as `/kaggle/working/`.

---

## 13. Free GitHub + Streamlit Deployment

No Kaggle runtime, ngrok, Cloudflare Tunnel, or local Python server is required for the final hosted app.

### Step 1 — Create a GitHub repository

Create a repository such as:

```text
PneumoVision-AI
```

### Step 2 — Upload four files

Upload:

```text
app.py
pneumonia_efficientnetb0.keras
requirements.txt
README.md
```

### Step 3 — Open Streamlit Community Cloud

Connect your GitHub account and create a new app.

Select:

```text
Repository: PneumoVision-AI
Branch: main
Main file: app.py
```

Then deploy.

The hosted app loads `pneumonia_efficientnetb0.keras` from the repository rather than trying to access Kaggle's filesystem.

---

## 14. Why the Dataset Is Not in the Repository

Training requires the complete image dataset, but inference does not.

For deployment:

```text
Uploaded X-ray
      ↓
Saved trained model
      ↓
Prediction
      ↓
Grad-CAM
```

Therefore, keeping the dataset out of GitHub makes the deployment package much smaller and simpler.

---

## 15. Important Limitations

### Not clinically validated

The model has not been clinically validated or approved as a medical device.

### False predictions are possible

The reported test accuracy was 86.38%, so the model does not classify every test image correctly.

### False negatives are possible

Although pneumonia recall was high in the reported test evaluation, some pneumonia images were still classified as normal.

### Dataset distribution matters

Performance can change for images from different hospitals, scanners, acquisition protocols, patient populations, or disease distributions.

### Confidence is not certainty

A high probability is a model output, not a medical certainty.

### Grad-CAM is not lesion segmentation

The highlighted region represents model attention and is not proof of pneumonia, a lesion, or a specific anatomical abnormality.

---

## 16. Educational Value

This project demonstrates:

- Computer vision
- Convolutional neural networks
- Transfer learning
- EfficientNet
- Fine-tuning
- Binary classification
- Class imbalance handling
- Class weighting
- Model evaluation
- ROC-AUC
- Precision and recall
- Confusion matrices
- Explainable AI
- Grad-CAM
- Streamlit
- GitHub deployment

---

## 17. Final Summary

**Project:** PneumoVision AI  
**Task:** Chest X-ray pneumonia classification  
**Model:** Fine-tuned EfficientNetB0  
**Input:** 224 × 224 RGB  
**Classes:** NORMAL, PNEUMONIA  
**Test Accuracy:** 86.38%  
**Test ROC-AUC:** 0.95035  
**Pneumonia Recall:** 97%  
**Explainability:** Grad-CAM using `top_conv`  
**Interface:** Streamlit  
**Deployment:** GitHub + Streamlit Community Cloud

---

## ⚠️ Final Medical Disclaimer

PneumoVision AI is an educational machine-learning project. It is not a clinically validated diagnostic system and must not be used as a substitute for a radiologist or other qualified healthcare professional. Predictions and Grad-CAM visualizations can be incorrect.
