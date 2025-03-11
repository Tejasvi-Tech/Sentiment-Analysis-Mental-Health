!pip install pandas scikit-learn nltk
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("/content/cleaned_mental_health_dataset.csv")

# Initialize preprocessing tools
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Download the 'punkt_tab' data package
nltk.download('punkt_tab') # This line was added to download the required data

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Define preprocessing function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Stopword removal & Lemmatization
    return " ".join(tokens)

# Apply preprocessing
df["cleaned_text"] = df["post_text"].apply(preprocess_text)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]  # Assuming "label" is the target column (0 = Non-Depressed, 1 = Depressed)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create the 'models' directory if it doesn't exist
os.makedirs("../models", exist_ok=True)  



# Save the trained model
import pickle
with open("../models/logistic_regression.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model training complete. Saved logistic_regression.pkl.")
