# 🚢 Titanic Survival Prediction
> A machine learning project to predict passenger survival on the Titanic, achieving 83.24% validation accuracy.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange)
![NumPy](https://img.shields.io/badge/NumPy-blueviolet)

## 1. Project Overview

This project uses machine learning to predict which passengers survived the sinking of the Titanic.

Using the provided passenger data, I performed a complete data science workflow:
1.  **Data Cleaning:** Handled missing values using `KNNImputer`.
2.  **Feature Engineering:** Extracted and created new features from `Name`, `Ticket`, `Cabin`, and family size columns.
3.  **Model Training:** Trained a `RandomForestClassifier` model.
4.  **Evaluation:** Achieved a **83.24% accuracy** on a 20% validation split.

## 2. Tech Stack

* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For feature preprocessing, imputation, modeling, and evaluation.

## 3. Dataset

The data was provided by the [Kaggle "Titanic - Machine Learning from Disaster" competition](https://www.kaggle.com/c/titanic).

* `train.csv`: The training set used to build the model.
* `test.csv`: The test set for which predictions are made.
* `titanic_predictions.csv`: The final output file with predictions.

**Note:** Per the `.gitignore` file, the `.csv` data files are not uploaded to this repository. You must download them directly from the Kaggle link above.

## 4. How to Run This Project

1.  Clone this repository:
    ```bash
    git clone https://github.com/Shravan-Solanki/Titanic-Survival-Prediction.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Titanic-Survival-Prediction
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/titanic/data) and place them in the root of the project folder.

5.  Run the Python script:
    ```bash
    python titanic.py
    ```
    or

    Run the Python notebook:
    ```bash
    python titanic.ipynb
    ```
    This will print the validation accuracy to the console and generate the `titanic_predictions.csv` submission file.

## 5. Methodology & Workflow

My approach focused heavily on data cleaning and feature engineering to prepare the data for the model.

### Data Cleaning & Preprocessing

1.  **Missing Values Imputation:** Instead of simple mean/median, I used `sklearn.impute.KNNImputer` (with `n_neighbors=13`) to fill missing values in `Age`, `Fare`, `Deck`, and `Embarked`. This uses feature similarity to make more accurate estimates.
2.  **Label Encoding:** Before imputation, `Deck` (from `Cabin`) and `Embarked` were label-encoded.
3.  **Scaling:** `Age` and `Fare` features were scaled using `StandardScaler`.

### Feature Engineering

I created several new features to help the model find patterns:

* **`Title`:** Extracted from the `Name` column (e.g., 'Mr', 'Miss', 'Mrs') and grouped rare titles into a single 'Rare' category.
* **`TicketPrefix`:** Extracted from the `Ticket` column. Prefixes with low frequency were grouped into a 'Rare' category.
* **`TicketGroupSize`:** Calculated the number of passengers traveling on the same ticket number.
* **`Deck`:** Extracted the first letter of the `Cabin` (A, B, C, etc.) as a proxy for the passenger's location on the ship.
* **`FamilySize`:** A new feature created by combining `SibSp` (siblings/spouses) and `Parch` (parents/children).
* **`IsAlone`:** A boolean feature (1 or 0) derived from `FamilySize`.

Finally, all categorical features (`Sex`, `Title`, `TicketPrefix`, `Deck`, `Embarked`) were converted into numerical dummy variables.

## 6. Model Selection & Results

* **Model:** A `RandomForestClassifier` was chosen for its strong performance and robustness.
* **Validation:** The training data was split (80/20) to create a local validation set.
* **Result:** The model achieved a **Validation Accuracy of 0.8324 (83.24%)**.

This result was used to generate the final `titanic_predictions.csv` file for the test set.

## 7. License
This project is licensed under the MIT License.