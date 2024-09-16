import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('city_day.csv')

# Clean column names
data.columns = [col.strip() for col in data.columns]

# Handle missing values
data.ffill(inplace=True)  # Forward fill to handle missing values

# Calculate AQI (simplified example)
data['AQI'] = data[['SO2', 'NO2', 'RSPM/PM10', 'SPM']].mean(axis=1)

# Classify AQI into categories
def classify_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 200:
        return 'Poor'
    elif aqi <= 300:
        return 'Very Poor'
    else:
        return 'Severe'

data['AQI_Category'] = data['AQI'].apply(classify_aqi)

# Prepare features and labels
features = data[['SO2', 'NO2', 'RSPM/PM10', 'SPM']]
labels = data['AQI_Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
