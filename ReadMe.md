```markdown
# 🧵 Surface Defect Detection Demo — Textile Production

This repository contains a **supervised deep-learning demo** for pixel-level defect segmentation in industrial textile images.  
A U-Net model, trained on manually labeled data, predicts surface anomalies. The demo ships with a modern Flask web app where users can upload an image and instantly view the predicted defect mask side-by-side with the original.

---

## 🎯 Demo Highlights

| Feature           | Details                                                                      |
|-------------------|------------------------------------------------------------------------------|
| **Architecture**  | U-Net (encoder–decoder with skip connections)                                |
| **Learning type** | **Supervised** segmentation — model trained on expert-annotated masks        |
| **Use case**      | Inline quality control / surface-defect detection in textile production      |
| **Web UI**        | Drag-and-drop upload, responsive layout, original & mask displayed together |

---

## 🧠 Model Training Summary

1. **Dataset**   DAGM 2007 (Class 1) images resized to 256 × 256 grayscale
2. **Labels**    Binary defect masks (1 = defect, 0 = background)
3. **Training**  U-Net, 50 epochs, Adam (lr 1e-4), BCELoss
4. **Validation** 80 / 20 split; best weights saved as `unet_dagm_class1.pth`

> **Note:** This is supervised ML — not a generic “AI agent.”
> The model learns strictly from labeled examples.

---

## ⚙️ Adapting to Your Production Line

* **Different materials?** Fine-tune with your own labeled dataset.
* **Other defect types?** Retrain with additional mask channels.
* **Edge deployment?** Containerized builds available for on-prem devices.

For professional integration or custom training, contact us: **[info@sagel-ai.com](mailto:info@sagel-ai.com)**

---

## ⚖️ License

MIT License © 2025 Your Company Name

```
```
