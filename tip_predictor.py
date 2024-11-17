import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


tips_data = pd.read_csv('C:/Users/kirti/Downloads/Project SCP/Waiter-Tips-Prediction-main/Waiter-Tips-Prediction-main/tips.csv')


print("Dataset Head:")
print(tips_data.head())
print("\nDataset Info:")
print(tips_data.info())
print("\nDataset Description:")
print(tips_data.describe())


print("\nMissing Values in Each Column:")
print(tips_data.isnull().sum())


print("\nUnique Days:")
print(tips_data['day'].unique())


plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips_data, hue='day', palette='viridis')
plt.title('Total Bill vs. Tip Amount', fontsize=16)
plt.xlabel('Total Bill', fontsize=14)
plt.ylabel('Tip Amount', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(tips_data['tip'], bins=20, kde=True, color='blue')
plt.title('Distribution of Tips', fontsize=16)
plt.xlabel('Tip Amount', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()


average_tips_by_day = tips_data.groupby('day')['tip'].mean().reset_index()
print("\nAverage Tip by Day:")
print(average_tips_by_day)

plt.figure(figsize=(10, 6))
sns.barplot(x='day', y='tip', data=average_tips_by_day, palette='pastel')
plt.title('Average Tip Amount by Day', fontsize=16)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Average Tip Amount', fontsize=14)
plt.tight_layout()
plt.show()


X = tips_data[['total_bill', 'size']]
y = tips_data['tip']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')  # line of equality
plt.title('Actual vs Predicted Tips', fontsize=16)
plt.xlabel('Actual Tips', fontsize=14)
plt.ylabel('Predicted Tips', fontsize=14)
plt.tight_layout()
plt.show()