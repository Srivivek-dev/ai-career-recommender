import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# -------------------------------------------------------
# 1. Create dummy career dataset (you can later replace)
# -------------------------------------------------------

data = {
    "skills": [
        "python machine learning ai numpy pandas",
        "html css javascript react frontend",
        "networking security linux firewalls cybersecurity",
        "sql database mysql oracle dbms",
        "java spring backend api development",
        "ui ux design figma wireframes prototyping",
        "c c++ dsa algorithms competitive programming",
        "cloud aws devops docker kubernetes ci cd"
    ],
    "career": [
        "AI/ML Engineer",
        "Frontend Developer",
        "Cybersecurity Analyst",
        "Database Engineer",
        "Backend Developer",
        "UI/UX Designer",
        "Software Developer",
        "Cloud/DevOps Engineer"
    ]
}

df = pd.DataFrame(data)

# -------------------------------------------------------
# 2. Vectorizer (Skill → Numeric)
# -------------------------------------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["skills"])

# -------------------------------------------------------
# 3. Random Forest Classifier (Main predictor)
# -------------------------------------------------------

classifier = RandomForestClassifier()
classifier.fit(X, df["career"])

# -------------------------------------------------------
# 4. KMeans clustering (Group similar skills)
# -------------------------------------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# -------------------------------------------------------
# 5. Role-to-skill map (For recommendations)
# -------------------------------------------------------

role_skill_map = {
    "AI/ML Engineer": ["python", "ml", "ai", "pandas", "numpy"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Cybersecurity Analyst": ["security", "linux", "networking"],
    "Database Engineer": ["sql", "mysql", "dbms"],
    "Backend Developer": ["java", "spring"],
    "UI/UX Designer": ["ui", "ux", "figma"],
    "Software Developer": ["c", "cpp", "dsa"],
    "Cloud/DevOps Engineer": ["aws", "docker", "kubernetes"]
}

# -------------------------------------------------------
# 6. Save all models to /models directory
# -------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(classifier, "models/career_classifier.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(role_skill_map, "models/role_skill_map.pkl")

print("✅ All model files created successfully in /models folder!")
