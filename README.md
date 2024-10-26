# Fraud Detection Using Decision Tree Classifier

## Project Overview

This project focuses on detecting fraudulent transactions based on historical transaction data. Using a Decision Tree Classifier, the goal is to predict whether a given transaction is fraudulent or not.

## Features

- **Binary Classification**
- **Data Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Model Training and Evaluation**
- **Prediction Example**

## Requirements

- **Python 3.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Plotly**

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Tanjim-Islam/Fraud-Detection-Using-Decision-Tree-Classifier.git
    cd Fraud-Detection
    ```

2. **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn plotly
    ```

## Code Structure

1. **Exploratory Data Analysis (EDA):**
   - Visualizes the distribution of transaction types using a pie chart.

2. **Data Preprocessing:**
   - Encodes categorical variables (e.g., transaction type).
   - Splits data into features (`x`) and target labels (`y`).

3. **Model Training and Evaluation:**
   - Trains a Decision Tree Classifier.
   - Evaluates the model using the testing set.

4. **Example Prediction:**
   - Uses the trained model to predict whether a sample transaction is fraudulent.

## Results

### Model Accuracy:

- **Model Score (Test Data)**: **99.97%**

### Example Prediction:

Below is a sample prediction using the trained Decision Tree Classifier model:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

features = np.array([[4, 9000.60, 9000.60, 0.0]])

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain) 

# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
```
