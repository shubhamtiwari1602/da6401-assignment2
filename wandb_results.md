# W&B Experiment Results — DA6401 Assignment 2

**Project:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2

---

## 2.1 The Regularization Effect of Dropout (BatchNorm baseline)

| Run | Final Train Loss | Final Val Loss |
|-----|-----------------|----------------|
| `baseline_with_bn` | 8.1188 | 11.9127 |

**W&B Run:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/6ih9vr1j

---

## 2.2 Internal Dynamics — Dropout Comparison

| Run | Dropout p | Final Train Loss | Final Val Loss |
|-----|-----------|-----------------|----------------|
| `dropout_0.0` | 0.0 (No Dropout) | 6.9676 | 10.7657 |
| `dropout_0.2` | 0.2 | 7.5399 | 10.3498 |
| `dropout_0.5` | 0.5 | 8.0474 | 11.6164 |

**W&B Runs:**
- dropout_0.0: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/afv9g0lz
- dropout_0.2: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/b3ydlmct
- dropout_0.5: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/f2w2hump

**Key Observation:** p=0.2 achieved the best validation loss (10.35), while p=0.5 showed the largest train/val gap, indicating stronger regularization. No dropout overfits most aggressively (train: 6.97 but val: 10.77).

---

## 2.3 Transfer Learning Showdown

| Run | Strategy | Final Train Loss | Final Val Loss |
|-----|----------|-----------------|----------------|
| `tl_strict` | Frozen backbone | 9.1190 | 11.4441 |
| `tl_partial` | Partial fine-tune | 8.2481 | 11.3879 |
| `tl_full` | Full fine-tune | 8.3473 | 11.1436 |

**W&B Runs:**
- tl_strict: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/u25irnql
- tl_partial: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/keq1frgr
- tl_full: https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/vzrwprt3

**Key Observation:** Full fine-tuning (`tl_full`) achieved the best validation loss (11.14). Strict freezing was fastest per epoch (1.82 it/s vs 1.52 it/s) but converged to a higher loss floor due to the pretrained features not adapting to the task.

---

## 2.4 Feature Maps

- **W&B Run:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/hl3o7cdo
- First conv layer feature maps (16 filters): low-level edges and gradients
- Last conv layer feature maps (16 filters): high-level semantic shapes

---

## 2.5 Object Detection Bounding Box Table

- **W&B Run:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/cqsiiclg
- 10 test images with Green (GT) and Red (Pred) bounding boxes
- Confidence score + IoU shown per image

---

## 2.6 Segmentation Evaluation

| Metric | Score |
|--------|-------|
| Mean Pixel Accuracy | **0.863** |
| Mean Dice Score | **0.821** |

- **W&B Run:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/fxgeqtje
- 5 sample images with Original / GT Trimap / Predicted Mask columns

---

## 2.7 In-the-Wild Showcase

- **W&B Run:** https://wandb.ai/me22b196-indian-institute-of-technology-madras/DA6401-Assignment2/runs/36qjf826
- 3 test-set images used as in-the-wild proxies (Kaggle blocks external image downloads)
- Red bounding box predictions overlaid with breed class and confidence %
- Model predicted breed indices and confidence scores for each image

---

## Summary

- **Total W&B Runs:** 12 runs logged across all 7 experiments
- **Best Dropout:** p=0.2 (Val Loss: 10.35)
- **Best Transfer Strategy:** Full Fine-Tuning (Val Loss: 11.14)
- **Segmentation Pixel Accuracy:** 0.863
- **Segmentation Dice Score:** 0.821
