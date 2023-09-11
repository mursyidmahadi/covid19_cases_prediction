# COVID-19 Cases Prediction in Malaysia by using LSTM
This repository contains the case study of predicting covid-19 cases in malaysia by using LSTM neural network. This is done by building a deep learning model and it uses the past 30 days of cases data to help forecasting the number of cases for next day.

## Task Implementation

## 1. Exploratory Data Analysis
- Analyze the cases_new column and plot a line plot for time series problem.

## 2. Data Cleaning
- Change non-numeric values such as '' to null values with to_numeric.
- df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
- Interpolating missing data
- df['cases_new'] = df['cases_new'].interpolate(method='polynomial', order=2)
- Select cases_new column as data for feature and target variable.
- data = df['cases_new'].values

## 3. Data Preprocessing
2 steps were taken for data preprocessing
- Use Min-Max Scaler for data scaling.
- Set window size to 30 for past 30 days of data as feature variable and target variable

## 4. Model Development
- A model architecture is used for guidance.
- Set LSTM neurons to 64 as limit to prevent from brute forcingthe problem.
- Test data is 20# and rest is allocated to training data.
- Set batch size to 64 with 600 epochs.
- Mean Absolute Error Percentage and Mean Squared Error were used as metrics.
- Mean Absolute Error Function is the loss function
- No early stopping implemented.

## 5. Results
- The results for both training loss and training mse were taken.
- Compare it with the validation loss and validation mse.

## 6. Model Performance
- The objective for this case study is to achieve Mean Absolute Percentage Error of below 10%.
- The model developed achieve MAPE (Mean Absolute Percentage Error) of 5.17%.
- The model is success.
