# To Create multiple ROC curve for the Student data
# https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Load the dataset
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\Student_performance_data _.csv")

# Data overview
print(df.isnull().sum())
print(df.info())

# Prepare features and target variable
X = df.drop(['GradeClass'], axis=1)
y = df['GradeClass']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print model performance
print('Test Score:', model.score(X_test, y_test))
print('Train Score:', model.score(X_train, y_train))

# Display Confusion Matrix
y_pred = model.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=model.classes_).plot()
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Plot ROC Curve for each class
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)


# Plot heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Multi-class ROC AUC Curve
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(8, 8)):
    y_score = clf.decision_function(X_test)
    
    fpr, tpr, roc_auc = {}, {}, {}
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristics')
    
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC Curve (area = {roc_auc[i]:.2f} for label {i})')
    
    ax.legend(loc='best')
    ax.grid(alpha=0.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(model, X_test, y_test, n_classes=n_classes)

# Plot count of each class
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='GradeClass', data=df)
plt.title('Class Distribution')
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

# Display class distribution
print(df['GradeClass'].value_counts())
