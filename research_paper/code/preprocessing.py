import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab') 

# Load dataset 
df = pd.read_csv('/content/mental-health-dataset.csv.zip')  
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Stopword removal & Lemmatization
    return " ".join(tokens)

# Check dataset column names
print(df.columns)  

# Replace "text_column" with the correct column name
df["cleaned_text"] = df["post_text"].apply(preprocess_text)  # UPDATE with actual column name!

# Save the cleaned dataset
df.to_csv('/content/cleaned_mental_health_dataset.csv', index=False)

print("Preprocessing complete. Cleaned dataset saved.")

