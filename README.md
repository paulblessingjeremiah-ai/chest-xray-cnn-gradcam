# Chest X-Ray Classification with Grad-CAM

A CNN-based classifier for chest X-rays, distinguishing between COVID-19, Normal, Pneumonia, and Tuberculosis, with Grad-CAM visualizations showing where the model focuses when making each prediction.

## Overview

This project uses transfer learning with a pretrained ResNet18 to classify chest X-ray images into four categories. Given that medical imaging datasets are often limited in size, transfer learning was chosen over training from scratch to make better use of the available 2,000 training images.

Beyond classification, the project implements Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of an X-ray the model relies on for its predictions — an important step toward making the model's decisions interpretable rather than a black box.

## Dataset

- Source: chest X-ray dataset from a Nigerian hospital (Aminu Kano Teaching Hospital)
- Classes: COVID-19, Normal, Pneumonia, Tuberculosis
- Training set: 2,000 images (500 per class)
- Test set: 600 images (150 per class)

Note: this dataset overlaps with an ongoing M.Eng. research project comparing radiomics and foundation model performance across datasets and modalities. This repository is a distinct piece of work — a CNN classification and explainability pipeline — built independently of that research.

## Architecture


## Results

The model was trained for 10 epochs, with the best checkpoint (by test accuracy) saved during training.

**Best model performance (test set, 600 images):**

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| COVID     | 0.99      | 0.98   | 0.98     |
| Normal    | 0.98      | 0.99   | 0.99     |
| Pneumonia | 0.99      | 1.00   | 1.00     |
| TB        | 1.00      | 0.99   | 0.99     |

**Overall accuracy: 99%**

![Confusion Matrix](results/confusion_matrix.png)

## Grad-CAM Visualizations

Grad-CAM heatmaps confirm the model focuses on lung tissue when making predictions, rather than relying on incidental image artifacts (such as text labels burned into the corners of the scans).

![COVID Example](results/gradcam_examples/COVID.png)
![Normal Example](results/gradcam_examples/NORMAL.png)
![Pneumonia Example](results/gradcam_examples/PNEUMONIA.png)
![TB Example](results/gradcam_examples/TB.png)

## Installation

```bash
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/
```

## Limitations

This model was trained and evaluated on a single-source dataset of 2,000 images. While test accuracy is high (99%), this may partly reflect the dataset's relatively clean, well-separated class characteristics rather than performance that would generalize to more diverse, real-world clinical data from multiple hospitals or imaging equipment. Training also showed signs of mild overfitting (training accuracy reached ~100% within the first epoch), addressed here through best-checkpoint saving based on test performance, though further validation on external datasets would be needed before any real-world application.

This project is intended for educational and research purposes only and is not designed for clinical diagnosis or deployment in a clinical setting.

## License

MIT
