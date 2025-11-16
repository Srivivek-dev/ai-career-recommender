# generate_sample_dataset.py
import pandas as pd
import os

rows = [
    {"skills_text":"python pandas numpy matplotlib", "role":"Data Analyst"},
    {"skills_text":"python pytorch tensorflow deep learning", "role":"ML Engineer"},
    {"skills_text":"html css javascript react", "role":"Frontend Developer"},
    {"skills_text":"java spring sql aws", "role":"Backend Developer"},
    {"skills_text":"python nlp transformers", "role":"NLP Engineer"},
    {"skills_text":"docker kubernetes aws devops", "role":"DevOps Engineer"},
]

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_dataset.csv", index=False)

print("Dataset created.")
