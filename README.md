# PyTorch Assignment: Wine Variety Classification

## Overview
This project uses a PyTorch neural network to classify samples from Scikit-learn's Wine dataset into three wine classes.

The notebook adapts the binary-classification workflow covered in PyTorch class to a three-class classification problem.

## Dataset
The dataset is loaded using `sklearn.datasets.load_wine`.

- Total samples: 178
- Input features: 13
- Target classes: 3 (`class_0`, `class_1`, and `class_2`)

## Methodology
The notebook follows these steps:

1. Import the required libraries.
2. Load the Wine dataset and separate the input features from the target labels.
3. Split the data into 80% training data and 20% testing data, using stratified sampling.
4. Standardize the features using `StandardScaler`, fitted on the training data.
5. Convert the features and labels into PyTorch tensors.
6. Create a `DataLoader` with a batch size of 32 and shuffled training batches.
7. Define a neural network with hidden layers of 32 and 16 units, ReLU activation, and dropout of 0.2.
8. Configure the model with `CrossEntropyLoss` and the Adam optimizer, using a learning rate of 0.001.
9. Train the model for 60 epochs and print the loss for each epoch.
10. Evaluate the model using test accuracy, a confusion matrix, and a classification report.

## Environment and Hardware
The assignment was executed in Google Colab with the following environment:

- PyTorch version: `2.11.0+cu130`
- CUDA available: `True`
- GPU: NVIDIA Tesla T4
- CUDA version: `13.0`

These details reflect the runtime reported during execution of the notebook.

## Results
The recorded notebook output shows:

- Test accuracy: **97.22%**
- Test samples: 36
- Correct predictions: 35 out of 36

The confusion matrix and classification report are also included in the notebook.

## Tools and Libraries
- Python
- PyTorch
- NumPy
- Scikit-learn
- Google Colab

## Repository Contents
- `Wine_Variety_(PyTorch_Assignment).ipynb` — the notebook containing the implementation, explanations, training output, and evaluation results.

## How to Run
1. Open the notebook in Google Colab.
2. Run the cells sequentially from top to bottom.
3. Review the training loss and evaluation results.

To use a GPU in Google Colab, select a GPU runtime if one is available. Otherwise, the notebook can run on the CPU.

## Conclusion
This assignment demonstrates how to build, train, and evaluate a neural network in PyTorch for a three-class classification task using Scikit-learn's Wine dataset.
