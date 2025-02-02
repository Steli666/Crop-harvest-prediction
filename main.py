import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("crop_yield.csv")
columns_features = ['Crop', 'Region', 'Weather_Condition', 'Soil_Type', 'Fertilizer_Used', 'Irrigation_Used']

pd.pivot_table(df, values='Yield_tons_per_hectare', index='Region', columns='Crop', aggfunc='sum')
pd.pivot_table(df, values='Temperature_Celsius', index='Region', columns='Crop', aggfunc=np.mean)
pd.pivot_table(df, values='Rainfall_mm', index='Region', columns='Crop', aggfunc=np.mean)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_features = encoder.fit_transform(df[columns_features])

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns_features))

df_final = pd.concat([df.drop(columns=columns_features), encoded_df], axis=1)

df_final.head()
correlation_matrix = df_final.corr()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()
from sklearn.model_selection import train_test_split

X = df_final.drop(columns=['Yield_tons_per_hectare'])
y = df_final['Yield_tons_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
accuracy = r2_score(y_test, y_pred)

print(mse)
print(accuracy*100)