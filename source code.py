# tree_species_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Sample data: Replace this with real-world tree species dataset
data = {
    'leaf_length': [5.1, 7.0, 6.3, 4.9, 6.1, 5.5, 7.6],
    'leaf_width': [3.5, 3.2, 3.3, 3.1, 2.9, 2.4, 3.0],
    'bark_texture': [1, 2, 2, 1, 3, 2, 3],  # 1 = smooth, 2 = rough, 3 = flaky
    'tree_species': ['Oak', 'Maple', 'Maple', 'Oak', 'Pine', 'Pine', 'Pine']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['leaf_length', 'leaf_width', 'bark_texture']]
y = df['tree_species']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
