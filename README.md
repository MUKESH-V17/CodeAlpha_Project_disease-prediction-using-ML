# Disease Prediction Using Symptoms

This repository contains a machine learning model that predicts diseases based on a set of symptoms. The model utilizes multiple machine learning algorithms to classify the disease correctly based on input symptoms.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to develop a machine learning model that can predict various diseases from a set of symptoms. The dataset consists of medical symptoms that are commonly observed in patients and the corresponding disease labels. The machine learning algorithms used in this project include:
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes Classifier

## Dataset
The dataset used in this project contains the following columns:
- **Symptoms**: A list of symptoms experienced by a patient.
- **Prognosis**: The predicted disease label for the given symptoms.

You can find the dataset in the `data/` directory. The dataset has been preprocessed for handling categorical features and missing values.

### Example of columns:
1. Itching
2. Skin rash
3. Nodal skin eruptions
4. Continuous sneezing
5. Shivering
6. Chills
7. Joint pain
8. Stomach pain
9. Acidity
10. Ulcers on tongue
... (Add more features based on your dataset)

## Technologies Used
- **Python**: Programming language used for implementing the machine learning models.
- **scikit-learn**: Python library for machine learning models.
- **pandas**: Data manipulation and preprocessing.
- **matplotlib** and **seaborn**: Data visualization.
- **Jupyter Notebook**: Development environment used for code execution.

## Installation
To get started with this project, you need to clone this repository and install the required dependencies.


## Requirements File (requirements.txt)
Here is an example of the required libraries:

numpy==1.24.0

pandas==1.5.0

scikit-learn==1.2.0

matplotlib==3.6.0

seaborn==0.11.2

Modeling Process
In this project, we perform the following steps:

**Data Preprocessing:**
Handle missing values (if any).
Encode categorical variables using Label Encoding.

**Train-Test Split:**
Split the dataset into training and testing sets using train_test_split from scikit-learn.

## Model Training:

**We train three different classifiers:**
Decision Tree Classifier
Random Forest Classifier
Naive Bayes Classifier

## Model Evaluation:

**After training the models, we evaluate them using metrics such as:**
1.)Accuracy
2.)Precision
3.)Recall
4.)F1-Score
5.)ROC-AUC
Hyperparameter Tuning (Optional):

Fine-tuning models using grid search (optional, depending on the further experimentation).

## Results
The results of each classifier are evaluated based on accuracy, precision, recall, and F1-score. Here are the results for each model (this part can be filled with actual numbers based on the results from your model evaluation):

### Decision Tree
1.Accuracy: 0.85

2.Precision: 0.84

3.Recall: 0.86

4.F1-Score: 0.85

5.ROC AUC: 0.87

### Random Forest
1.Accuracy: 0.90

2.Precision: 0.88

3.Recall: 0.89

4.F1-Score: 0.88

5.ROC AUC: 0.92

### Naive Bayes
1.Accuracy: 0.80

2.Precision: 0.78

3.Recall: 0.81

4.F1-Score: 0.79

5.ROC AUC: 0.83

### Usage
1.To use this model to predict the disease based on the input symptoms

2.Prepare your feature data in the same format as the dataset (i.e., symptom columns as binary features).
Use the trained model to predict the disease.
