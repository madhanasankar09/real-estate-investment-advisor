import pandas as pd
import mlflow
import mlflow.sklearn


df = pd.read_csv("india_housing_prices.csv")

print(df.head())
print(df.info())


import pandas as pd

df = pd.read_csv("india_housing_prices.csv")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df.fillna(method='ffill', inplace=True)


print("Data cleaned successfully")

df.to_csv("cleaned_data.csv", index=False)
print("Cleaned dataset saved successfully")



# Price per sqft
df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']

# Age of property
df['Age_of_Property'] = 2025 - df['Year_Built']

# Good investment label
median = df['Price_per_SqFt'].median()

df['Good_Investment'] = df['Price_per_SqFt'].apply(
    lambda x: 1 if x < median else 0
)

print(df.head())


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("Encoding done")

df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1.08 ** 5)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Features
X = df[['Size_in_SqFt','BHK','Price_in_Lakhs']]


from xgboost import XGBClassifier, XGBRegressor

# -------- Classification --------
y1 = df['Good_Investment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y1, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
print("XGBoost classification trained")

mlflow.start_run(run_name="XGBoost_Classifier")

xgb_clf.fit(X_train, y_train)

accuracy = xgb_clf.score(X_test, y_test)

mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(xgb_clf, "xgb_classifier")

mlflow.end_run()

print("Classification model trained")

# -------- Regression --------
y2 = df['Future_Price_5Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y2, test_size=0.2, random_state=42
)

reg = RandomForestRegressor()
reg.fit(X_train, y_train)

xgb_reg = XGBRegressor()
xgb_reg.fit(X_train, y_train)
print("XGBoost regression trained")

mlflow.start_run(run_name="XGBoost_Regressor")

xgb_reg.fit(X_train, y_train)

score = xgb_reg.score(X_test, y_test)

mlflow.log_metric("r2_score", score)
mlflow.sklearn.log_model(xgb_reg, "xgb_regressor")

mlflow.end_run()


print("Regression model trained")

import pickle

pickle.dump(clf, open("clf.pkl","wb"))
pickle.dump(reg, open("reg.pkl","wb"))

print("Models saved successfully")


