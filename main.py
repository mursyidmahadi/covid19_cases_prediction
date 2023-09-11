# %% Import
import pandas as pd
import os

from module import eda, plot_data, data_mms, model_archi, model_train, model_save, predict_score, predict_plot, test_prepare

from sklearn.model_selection import train_test_split

WINDOW_SIZE = 30

TRAIN_PATH = os.path.join(os.getcwd(), 'data_set', 'cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(), 'data_set', 'cases_malaysia_test.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')
MODEL_PNG_PATH = os.path.join(os.getcwd(), 'model', 'model.png')
PKL_PATH = os.path.join(os.getcwd(), 'model', 'mms.pkl')

if not os.path.exists(os.path.join(os.getcwd(), 'model')):
    os.makedirs(os.path.join(os.getcwd(), 'model'))

# %% Step 1: Data Loading
df = pd.read_csv(TRAIN_PATH)

# %% Step 2: EDA
eda(df)

# %% Step 3: Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
print(df['cases_new'].isna().sum())
plot_data(df)

# %%
df['cases_new'] = df['cases_new'].interpolate(method='polynomial', order=2)
plot_data(df)

# %% Step 4: Features Selection
data = df['cases_new'].values

# %% Step 5: Data Preprocessing
mms, X, y = data_mms(WINDOW_SIZE, data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% Step 6: Model Development
model = model_archi(X_train, MODEL_PNG_PATH)
hist, model = model_train(model, X_train, y_train, X_test, y_test)

# %% Step 7: Model Analysis
y_pred = model.predict(X_test)
predict_score(y_test, y_pred)

# %% Step 8: Model Deployment
X_actual, y_actual = test_prepare(TEST_PATH, WINDOW_SIZE, df, data, mms)

# %% 
# Model Predictions
y_pred_actual = model.predict(X_actual)
predict_plot(mms, y_actual, y_pred_actual)
predict_score(y_actual, y_pred_actual)

# %%
# Model Saving
model_save(MODEL_PATH, PKL_PATH, model, mms)

# %%