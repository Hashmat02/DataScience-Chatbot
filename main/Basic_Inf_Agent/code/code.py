from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv('df.csv')

df = df.dropna()
df = df.select_dtypes(include=['int64', 'float64'])

X = df.drop('Fare', axis=1)
y = df['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
df.to_csv('./df.csv', index=False)