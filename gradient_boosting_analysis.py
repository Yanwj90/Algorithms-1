# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset (replace with your dataset path if needed)
df = pd.read_csv('/kaggle/input/positive-and-negative-test-cases/Labelled_Test_Cases.csv', encoding='latin1')
df = df[['v1', 'v2']]  # Keep only relevant columns

# Check for null values
df = df.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.3, random_state=42)

# Define a new algorithm: Gradient Boosting Classifier
new_classifier = GradientBoostingClassifier()

# Create a pipeline for text preprocessing and model training
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('transformer', TfidfTransformer()),
    ('classifier', new_classifier)
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"\nResults for Gradient Boosting Classifier:")
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Compare with other algorithms
results = {
    'Gradient Boosting': accuracy,
    'Multinomial Naive Bayes': 0.80,
    'Decision Tree': 0.93,
    'Random Forest': 0.91,
    'Support Vector Machine': 0.87,
    'Logistic Regression': 0.865
}

# Plot the comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Algorithm Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# Save the Gradient Boosting model for future use
import joblib
joblib.dump(model, 'gradient_boosting_model.pkl')

# Instructions for uploading to GitHub
# 1. Save this script as a .py file.
# 2. Create a new repository on GitHub.
# 3. Add this script and the dataset to the repository.
# 4. Use Git commands or GitHub Desktop to push the code to your repository.
# 5. Share the GitHub link to display the results.
