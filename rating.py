import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Program started...")

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\dataset.csv")
print("Dataset loaded:", df.shape)

# Select required columns
df = df[['Country Code', 'City', 'Cuisines',
         'Average Cost for two', 'Price range',
         'Has Table booking', 'Has Online delivery',
         'Aggregate rating', 'Rating text']]

# Rename columns
df.columns = ['country_code', 'city', 'cuisines',
              'average_cost_for_two', 'price_range',
              'has_table_booking', 'has_online_delivery',
              'aggregate_rating', 'rating_text']

# Clean data
df.dropna(inplace=True)
df = df[df['aggregate_rating'] > 0]

print("After cleaning:", df.shape)

# ===============================
# 📊 DATASET-BASED GRAPHS
# ===============================

# 1️⃣ Rating Category Distribution
rating_counts = df['rating_text'].value_counts()

plt.figure()
rating_counts.plot(kind='bar')
plt.xlabel("Rating Category")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Rating Categories")
plt.show(block=True)

# 2️⃣ Price Range vs Rating Category
price_rating = df.groupby('price_range')['rating_text'].value_counts().unstack()

plt.figure()
price_rating.plot(kind='bar', stacked=True)
plt.xlabel("Price Range")
plt.ylabel("Count")
plt.title("Price Range vs Rating Category")
plt.show(block=True)

# ===============================
# 🤖 DATA PREPARATION
# ===============================

# Encode categorical columns
le = LabelEncoder()

df['city'] = le.fit_transform(df['city'])
df['cuisines'] = le.fit_transform(df['cuisines'])
df['has_table_booking'] = le.fit_transform(df['has_table_booking'])
df['has_online_delivery'] = le.fit_transform(df['has_online_delivery'])
df['rating_text'] = le.fit_transform(df['rating_text'])

# Features & Target
X = df.drop(['aggregate_rating', 'rating_text'], axis=1)
y = df['rating_text']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 🧠 MODEL TRAINING
# ===============================
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

print("Model trained")

# Predictions
y_pred = model.predict(X_test)

# ===============================
# 📈 MODEL EVALUATION
# ===============================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 📊 CONFUSION MATRIX GRAPH
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show(block=True)

# ===============================
# 📊 FEATURE IMPORTANCE
# ===============================
plt.figure()
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance for Rating Classification")
plt.show(block=True)

print("\nTask 3 completed successfully.")
