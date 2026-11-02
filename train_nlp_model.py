import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Generating synthetic e-commerce customer support dataset...")
np.random.seed(42)

# Sample ticket texts and their corresponding departments & urgency
tickets_data = [
    ("My order has not arrived yet and it is 5 days late!", "Shipping & Logistics", "High"),
    ("Where is my package? The tracking number isn't updating.", "Shipping & Logistics", "Medium"),
    ("The delivery driver threw my box over the fence and broke the item.", "Shipping & Logistics", "High"),
    
    ("I want to return this dress because it is too small. How do I get a refund?", "Returns & Refunds", "Medium"),
    ("I received a damaged product and need a replacement immediately.", "Returns & Refunds", "High"),
    ("How many days do I have left to return my order?", "Returns & Refunds", "Low"),
    
    ("My payment was deducted twice for the same order! Please refund one.", "Billing & Payments", "High"),
    ("I cannot apply my promo code at checkout. It says invalid.", "Billing & Payments", "Low"),
    ("Where can I download my tax invoice for this purchase?", "Billing & Payments", "Low"),
    
    ("Your mobile app crashes every time I click on my cart.", "Technical Support", "Medium"),
    ("I am locked out of my account and password reset email is not arriving.", "Technical Support", "High"),
    ("The website layout is broken on my browser.", "Technical Support", "Low")
]

# Expand dataset to make it robust for training
expanded_texts = []
expanded_depts = []
expanded_urgencies = []

for text, dept, urgency in tickets_data:
    expanded_texts.append(text)
    expanded_depts.append(dept)
    expanded_urgencies.append(urgency)
    
    # Add minor variations to expand dataset size
    expanded_texts.append(f"URGENT: {text}")
    expanded_depts.append(dept)
    expanded_urgencies.append("High" if urgency != "Low" else "Medium")
    
    expanded_texts.append(f"Hello support, {text.lower()}")
    expanded_depts.append(dept)
    expanded_urgencies.append(urgency)

df = pd.DataFrame({
    'Ticket_Text': expanded_texts,
    'Department': expanded_depts,
    'Urgency': expanded_urgencies
})

print(f"Dataset created with {len(df)} sample tickets.")

# Features (X) and Targets (y for Department, y for Urgency)
X = df['Ticket_Text']
y_dept = df['Department']
y_urgency = df['Urgency']

# Split dataset into training and testing sets
X_train, X_test, y_dept_train, y_dept_test, y_urg_train, y_urg_test = train_test_split(
    X, y_dept, y_urgency, test_size=0.2, random_state=42
)

# NLP Text Vectorization using TF-IDF (Term Frequency-Inverse Document Frequency)
print("Converting text complaints into numerical vectors using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Department Classification Model
print("Training Department Router Model (Random Forest)...")
dept_model = RandomForestClassifier(n_estimators=100, random_state=42)
dept_model.fit(X_train_vec, y_dept_train)

# Train Urgency Classification Model
print("Training Urgency Predictor Model (Random Forest)...")
urg_model = RandomForestClassifier(n_estimators=100, random_state=42)
urg_model.fit(X_train_vec, y_urg_train)

# Evaluate Models
dept_preds = dept_model.predict(X_test_vec)
print(f"Department Routing Accuracy: {accuracy_score(y_dept_test, dept_preds) * 100:.2f}%")

# Save vectorizer and trained models using pickle
with open('ticket_models.pkl', 'wb') as f:
    pickle.dump((vectorizer, dept_model, urg_model), f)

print("\nNLP models trained and saved successfully as 'ticket_models.pkl'!")