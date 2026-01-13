import csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Example dataset
df = pd.read_csv("D:\Sooraj\Project_Bromine_Concentration/app\data/test.csv", parse_dates=['Date'], dayfirst=True)
df = df.sort_values('Date')

# Features and target
X = df[['Brine_In', 'Brine_Out', 'Initial_Concentration_(gpl)']]
y = df['Final_Concentration_(gpl)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = XGBRegressor(
    n_estimators=200,       # number of boosting rounds
    learning_rate=0.1,      # step size shrinkage
    max_depth=4,            # depth of trees
    subsample=0.8,          # fraction of samples used per tree
    colsample_bytree=0.8,   # fraction of features used per tree
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
# y_pred = model.predict(X_test)

#[[220186, 174828, 0.26],[180910, 162424, 0.24],[160182, 162004, 0.23],[165361, 172824, 0.26],[181710, 163372, 0.22],[154243, 172836, 0.22],[208378, 164236, 0.21],[205635, 173020, 0.2]]
# Predict on new data
new_data = pd.read_csv("D:\Sooraj\Project_Bromine_Concentration/app\data/tester.csv")
new_data["predicted_final"] = model.predict(new_data)
new_data.to_csv("D:\Sooraj\Project_Bromine_Concentration/app\predictions.csv", index=False)

