# Multiple ROC curve on Student data

# Student Performance Data Analysis

This project involves analyzing the student performance dataset using Logistic Regression and evaluating the model with ROC curves and other metrics. The dataset used is available on Kaggle: [Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Analysis and Visualization](#analysis-and-visualization)

## Installation

Ensure you have the necessary Python libraries installed. You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

- **Load the Dataset**: Update the path to the CSV file in the script according to your local setup.

- **Run the Analysis**: Execute the Python script to load the data, train the logistic regression model, and generate various performance metrics.

- **View Results**:
  - The script will print the test and train scores of the model.
  - It will display a confusion matrix and classification report.
  - ROC curves for each class will be plotted to visualize the model’s performance.
  - A heatmap showing the correlation between features will be generated.
  - Class distribution will be visualized with a count plot.

## Analysis and Visualization

The script performs the following analyses and visualizations:

- **Data Overview**:
  - Prints information about missing values and dataset structure.

- **Data Preparation**:
  - Splits the dataset into training and testing sets.
  - Scales features using `StandardScaler`.

- **Model Training**:
  - Trains a `LogisticRegression` model on the training data.

- **Model Evaluation**:
  - Prints the accuracy scores for both training and testing datasets.
  - Displays a confusion matrix.
  - Prints a classification report.

- **ROC Curves**:
  - Plots ROC curves for each class to evaluate model performance.
  - Uses One-vs-Rest strategy for multi-class classification.

- **Feature Correlation**:
  - Generates a heatmap to show correlations between features.

- **Class Distribution**:
  - Plots the count of each class in the dataset.


