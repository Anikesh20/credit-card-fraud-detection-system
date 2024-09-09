
# Credit Card Fraud Detection

This project aims to build a machine learning model to detect fraudulent credit card transactions using a dataset containing transaction details. The system predicts whether a transaction is fraudulent or legitimate based on the input features. The project also includes a Streamlit-based web interface to provide an easy-to-use interface for predictions.

## Project Structure

- `app.py`: This file contains the Streamlit web app code that loads the trained model and provides a UI for users to input transaction details and get predictions.
- `credit_card_fraud_detection.ipynb`: The Jupyter Notebook used for data preprocessing, training, and evaluation of various machine learning models.
- `credit_card_fraud_detection.pkl`: This is the saved machine learning model used by the web app to predict fraud.
- `x_train_columns.pkl`: This file contains the column names of the training data after preprocessing, ensuring that new data is correctly aligned during prediction.

## Datasets

The dataset contains credit card transactions, with each transaction labeled as either fraud (`is_fraud=0`) or not fraud (`is_fraud=1`). The dataset is split into training and test sets, with features such as transaction amount (`amt`), merchant name, and city.

### Dataset Columns:
- `amt`: Transaction amount
- `merchant`: The merchant where the transaction took place
- `city`: The city where the transaction occurred
- `trans_date_trans_time`: Transaction date and time
- `is_fraud`: Target label indicating whether the transaction is fraudulent (1) or not (0)
- Dataset link https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Models Used

Three machine learning models were evaluated for this task:

1. **Logistic Regression**: A linear model used for binary classification.
2. **Decision Tree**: A tree-based model that splits data based on feature thresholds.
3. **Random Forest**: An ensemble of decision trees, providing more accurate results.

After training and testing, the best model was selected based on accuracy.

## Preprocessing

- Categorical features such as `merchant` and `city` were one-hot encoded.
- Numerical features were scaled using the `StandardScaler` to ensure all features have equal weight in model training.
- Features such as `first`, `last`, `street`, `zip`, and `dob` were removed, as they were not necessary for the prediction task.

## Model Training

The training process involved the following steps:

1. Preprocessing the data (scaling numerical values and encoding categorical values).
2. Training multiple machine learning models.
3. Selecting the best model based on accuracy.
4. Saving the model as `credit_card_fraud_detection.pkl` for future use in the web app.

## Streamlit Web App

The `app.py` file contains the code for a simple web interface using Streamlit. Users can input transaction details such as the amount, merchant, city, and transaction date. The web app then loads the trained model and predicts whether the transaction is fraudulent.

## Running the Web App

1. Install the necessary packages using `pip install -r requirements.txt`.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Input transaction details into the web app.
4. Get a prediction of whether the transaction is fraudulent or not.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook (`credit_card_fraud_detection.ipynb`) to see the data preprocessing, model training, and evaluation.
4. Run the Streamlit app (`app.py`) to interact with the model and make predictions.

## Conclusion

This project demonstrates how to build a credit card fraud detection system using machine learning. It includes data preprocessing, model training, and a user-friendly web interface for making predictions.

