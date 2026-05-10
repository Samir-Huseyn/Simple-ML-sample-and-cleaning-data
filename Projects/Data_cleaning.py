import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("dirty_cafe_sales.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())
print((df == "ERROR").sum())
print((df == "UNKNOWN").sum())
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")
df = df.replace("UNKNOWN", pd.NA)
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Price Per Unit"] = df["Price Per Unit"].fillna(df["Price Per Unit"].median())
df["Total Spent"] = df["Total Spent"].fillna(df["Total Spent"].median())
df["Item"] = df["Item"].fillna(df["Item"].mode()[0])
df["Payment Method"] = df["Payment Method"].fillna(df["Payment Method"].mode()[0])
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
df["month"] = df["Transaction Date"].dt.month
print(df.isnull().sum())
print(df.describe())
df = df.dropna(subset = ["Transaction Date"])
df["month"] = df["Transaction Date"].dt.month
print(df.isnull().sum())
print((df["Quantity"] * df["Price Per Unit"] == df["Total Spent"]).mean())

X = df[["Quantity", "Price Per Unit", "month"]]
y = df["Total Spent"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = mean_absolute_error(y_test,y_pred)
new_data = pd.DataFrame({
    "Quantity" : [3],
    "Price Per Unit" : [4],
    "month" : [7]
})
guess = model.predict(new_data)
print("Prediction:", guess[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(error)