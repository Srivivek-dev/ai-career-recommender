# career_model.py
import numpy as np

class CareerModel:
    def __init__(self, artifacts):
        self.vectorizer = artifacts['vectorizer']
        self.classifier = artifacts['classifier']
        self.kmeans = artifacts.get('kmeans')
        self.role_skill_map = artifacts.get('role_skill_map', {})

    def predict_roles_from_skilllist(self, skills):
        text = " ".join(skills)
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        top3 = np.argsort(probs)[::-1][:3]
        return [f"{classes[i]} (score={probs[i]:.2f})" for i in top3]

    def skill_score(self, skills):
        text = " ".join(skills)
        role = self.classifier.predict(self.vectorizer.transform([text]))[0]
        required = set(self.role_skill_map.get(role, []))
        if not required:
            return 50
        present = len(required.intersection(set(skills)))
        return 100 * present / len(required)

    def suggest_missing_skills(self, skills):
        text = " ".join(skills)
        role = self.classifier.predict(self.vectorizer.transform([text]))[0]
        required = set(self.role_skill_map.get(role, []))
        return list(required - set(skills))
