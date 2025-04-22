# Fraud-Detection-Using-Machine-Learning-and-Apache-Spark
This Jupyter Notebook (fraud_detection.ipynb) demonstrates batch fraud detection on credit card transactions using Apache Spark ML alongside Scikit-Learn. It covers data loading, preprocessing, model training, evaluation, and visualization of results.

# Dataset
We use the Kaggle Credit Card Fraud Detection dataset, which contains 284,807 transactions with only 492 fraud cases (0.17% of all transactions). Each record includes 30 features (V1–V28, Time, Amount) and a binary Class label (0 = legitimate, 1 = fraudulent).
Link-https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

# Installation
This project was developed and run on Kaggle Notebooks. No manual installation is required if you open and run it directly there.

# Technologies & Libraries
Apache Spark 3.3.0 (SparkSession, DataFrame API, Spark ML)

Python 3.8+

Scikit-Learn (SVM, model calibration, metrics)

TensorFlow / Keras (for any deep learning extensions)

NumPy & Pandas (data manipulation)

Matplotlib (visualization)

# Notebook Workflow
1. Spark Session Setup: Initialize Spark with required Kafka package (for streaming extension).

2. Data Loading: Read CSV into a Spark DataFrame.

3. Feature Engineering: Assemble features (V1–V28, Amount) and apply StandardScaler.

4. Train/Test Split: Convert to Pandas/NumPy arrays and split into training and testing sets.

5. Model Training:

   - Train an SVM with RBF kernel and class_weight='balanced' to handle class imbalance.

   - Optionally calibrate probabilities with CalibratedClassifierCV.

6. Evaluation:

  - Generate classification report (precision, recall, F1-score).

  - Plot confusion matrix and ROC curve with AUC.

7. Visualization: Display performance metrics and training curves.


