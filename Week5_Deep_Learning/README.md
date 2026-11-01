# Handwritten Digit Recognition Using Deep Learning

## Week 5 – Deep Learning Application in Data Science

A multi-class image classification project that recognizes handwritten digits (0–9) using the MNIST dataset and a neural network built with TensorFlow/Keras.

## Project Results

| Metric | Result |
|---|---:|
| Test accuracy | **97.86%** |
| Weighted precision | 0.9787 |
| Weighted recall | 0.9786 |
| Weighted F1-score | 0.9786 |
| Test loss | 0.0718 |
| Correct test predictions | 9,786 / 10,000 |
| Incorrect test predictions | 214 / 10,000 |

The model was trained for 10 epochs. The final training accuracy was 98.25%, and the final validation accuracy was 98.07%.

## Dataset

The project uses MNIST, a dataset of 70,000 grayscale handwritten digit images:

- 60,000 training images
- 10,000 test images
- Each image is 28 × 28 pixels
- Ten target classes: digits 0 through 9

The dataset is loaded directly through `tensorflow.keras.datasets.mnist`.

## Model Architecture

The Sequential model contains:

1. Input image: 28 × 28
2. Flatten: converts the image into 784 values
3. Dense layer: 128 neurons, ReLU activation
4. Dropout: 0.2
5. Dense layer: 64 neurons, ReLU activation
6. Output Dense layer: 10 neurons, Softmax activation

Total trainable parameters: **109,386**.

Compilation settings:
- Optimizer: Adam
- Loss: categorical cross-entropy
- Metric: accuracy

Training settings:
- Epochs: 10
- Batch size: 128
- Validation split: 10%

## Repository Structure

```text
Week5_Deep_Learning/
├── data/
├── models/
│   └── mnist_digit_recognition.keras
├── notebooks/
│   └── Week5_Handwritten_Digit_Recognition.ipynb
├── reports/
│   └── Week5_Handwritten_Digit_Recognition_Report.docx
├── screenshots/
│   ├── 01_mnist_samples.png
│   ├── 02_training_performance.png
│   ├── 03_confusion_matrix.png
│   ├── 04_sample_predictions.png
│   ├── 05_incorrect_predictions.png
│   ├── 06_classwise_performance.png
│   └── 07_saved_model_prediction.png
├── README.md
└── requirements.txt
```

The `data/` directory is reserved for any locally stored dataset files. MNIST is downloaded automatically by the Keras dataset loader when the notebook is first run.

## Requirements

- Python 3.10 or compatible version supported by the installed TensorFlow release
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository.
2. Open a terminal in the project root.
3. Install the dependencies using the command above.
4. Launch JupyterLab:

   ```bash
   jupyter lab
   ```

5. Open `notebooks/Week5_Handwritten_Digit_Recognition.ipynb`.
6. Run the notebook cells from top to bottom.

The notebook downloads MNIST through Keras, preprocesses the images and labels, builds the network, trains it, evaluates it, generates visualizations, and saves the model.

### Loading the Saved Model

From a Python session in the project root:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/mnist_digit_recognition.keras"
)
```

To predict a preprocessed image, provide a batch with shape `(1, 28, 28)` and pixel values normalized to the range 0–1.

## Key Analysis Findings

- The model correctly classified 9,786 of the 10,000 test images.
- The most frequent confusion was actual digit **4** predicted as **9** (18 cases).
- Other frequent confusions included 7→2 (11 cases) and 6→5 (11 cases).
- Digit 1 achieved an F1-score of 0.99 in the classification report.
- Digit 4 had the lowest class recall (0.96), while digit 9 had the lowest class precision (0.96).

## Visualizations

The `screenshots/` directory contains:
1. Sample images from MNIST.
2. Training and validation accuracy/loss curves.
3. Confusion matrix.
4. Sample predictions.
5. Incorrect predictions.
6. Class-wise precision, recall and F1-score.
7. Prediction after reloading the saved model.

## Limitations and Future Work

This model is designed for MNIST-style 28 × 28 grayscale images. Its performance on natural images or different handwriting sources has not been evaluated. Future work could compare this dense neural network with a CNN, add data augmentation, and test the model on user-drawn digits.

## Report

See `reports/Week5_Handwritten_Digit_Recognition_Report.docx` for the detailed project report, methodology, results, figures, error analysis, limitations and conclusion.

## License and Dataset Note

MNIST is a commonly used benchmark dataset distributed through Keras. Review the dataset's original terms and attribution requirements before redistributing dataset files. This repository contains the project code and model, not a separate copy of the MNIST dataset.
