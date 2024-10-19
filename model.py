# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'your_data.csv' with the actual file path)
df = pd.read_csv('crime_data.csv')

# Select relevant features (X) and target variable (y)
# Features: Age, Sex, Time of Occurrence, Area, Premise Description
X = df[['Vict Age', 'Vict Sex', 'TIME OCC', 'AREA', 'Premis Desc']]

# Target: Crime Description or Mode of Killing (you can adjust to your column name, e.g., 'Crm Cd Desc' or 'Weapon Desc')
y = df['Weapon Desc']  # or df['Weapon Desc']


# # Select relevant features (X) and target variable (y)
# # Features: Age, Sex, Time of Occurrence, Area, Premise Description
# X = df[['Vict Age', 'Vict Sex', 'TIME OCC', 'AREA', 'Premis Desc']]

# # Target: Crime Description or Mode of Killing (you can adjust to your column name, e.g., 'Crm Cd Desc' or 'Weapon Desc')
# y = df['Crm Cd Desc']  # or df['Weapon Desc']

# Preprocess the data
# Handle missing values if any
X = X.fillna('Unknown')
y = y.fillna('Unknown')

# Convert categorical variables to numeric using LabelEncoder
le_sex = LabelEncoder()
le_area = LabelEncoder()
le_premise = LabelEncoder()

X['Vict Sex'] = le_sex.fit_transform(X['Vict Sex'])
X['AREA'] = le_area.fit_transform(X['AREA'])
X['Premis Desc'] = le_premise.fit_transform(X['Premis Desc'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, max_depth=25, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = rf_model.feature_importances_
features = X.columns

# Display feature importance
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")
