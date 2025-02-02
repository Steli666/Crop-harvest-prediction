import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv("crop_yield.csv")

X = data.drop('Yield_tons_per_hectare', axis=1)
y = data['Yield_tons_per_hectare']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Region', 'Crop', 'Weather_Condition', 'Fertilizer_Used', 'Irrigation_Used', 'Soil_Type']),
        ('num', SimpleImputer(strategy='mean'), ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
#sample_input = pd.DataFrame({
#    'Region': ['North'],
#    'Crop': ['Wheat'],
#    'Weather_Condition': ['Sunny'],
#    'Rainfall_mm': [50.0],
#    'Temperature_Celsius': [25.0],
#    'Days_to_Harvest': [90],
#    'Soil_Type': [3],  # Loam
#    'Fertilizer_Used': [1],  # True
#    'Irrigation_Used': [1]   # True
#})

#predicted_yield = model.predict(sample_input)

#print(f"Predicted Yield (tons per hectare): {predicted_yield[0]:.2f}")
#joblib.dump(model, "crop_yield_model1.pkl")
#print("Model saved as 'crop_yield_model1.pkl'")
