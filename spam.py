import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load and prepare the data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0,'spam':1})

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'],
                                                    test_size=0.2, random_state=42)

# Create and fit the vectorizer
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train the model
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f} %")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save the model and vectorizer
joblib.dump(model, 'spam_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer have been saved successfully!")