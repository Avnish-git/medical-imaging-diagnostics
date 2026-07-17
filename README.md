# Medical Imaging Diagnostics
Pipeline for multi-label chest X-ray classification using TensorFlow.
![ROC Curves](ROC%20Curve%20Results.png)
### Per-Class AUC-ROC Performance (Test Set)

| Pathology | AUC Score |
| :--- | :--- |
| Atelectasis | 0.6474 |
| Cardiomegaly | 0.6067 |
| Effusion | 0.6989 |
| Infiltration | 0.6160 |
| Mass | 0.5738 |
| Nodule | 0.6124 |
| Pneumonia | 0.6562 |
| Pneumothorax | 0.6128 |
| Consolidation | 0.7771 |
| Edema | 0.7874 |
| Emphysema | 0.7880 |
| Fibrosis | 0.7003 |
| Pleural_Thickening | 0.7253 |
| **Hernia** | **0.9291** |

**Macro Average AUC:** 0.6951
### Next Steps & Limitations
* **Dataset Scale:** This pipeline was trained on a 5,606-image sample subset due to compute constraints. Training on the full NIH ChestX-ray14 dataset (112,120 images) would significantly improve per-class feature extraction.
* **Phase 2 Fine-Tuning:** The current baseline utilizes a frozen backbone. The next engineering step is to unfreeze the top layers of the backbone (DenseNet121/MobileNetV2) and train at a reduced learning rate (`1e-5`) to learn X-ray-specific edge features.
