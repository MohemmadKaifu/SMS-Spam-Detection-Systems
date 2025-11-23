import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a simple dummy dataset (normally you would load this from a CSV)
data = {
    'message': [
        'Win a free iPhone now!',         # Spam
        'Hey, are we meeting for lunch?', # Ham (Real)
        'Urgent! You have won $1000',     # Spam
        'Mom called, call her back',      # Ham
        'Free entry into a contest',      # Spam
        'Can you send me the notes?',     # Ham
        'Click here for a cash prize',    # Spam
        'Happy birthday! Have a great day'# Ham
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# 2. Preprocessing: Convert text to numbers (Vectorization)
# Machines can't read words, they read numbers. This turns words into a matrix of counts.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message']) # Features
y = df['label']                             # Target

# 3. Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train the Model (Using Naive Bayes - great for text)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Test the Model
print("--- Model Training Complete ---")

# Let's test it with a NEW message the model has never seen
new_message = ["Congratulations! You've won a free ticket"]
new_message_vectorized = vectorizer.transform(new_message)
prediction = model.predict(new_message_vectorized)

print(f"Testing message: '{new_message[0]}'")
print(f"Prediction: {prediction[0].upper()}")
