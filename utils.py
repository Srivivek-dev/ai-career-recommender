# utils.py
import joblib
import os
from typing import Dict, Any, List


def load_models(models_dir: str = "models") -> Dict[str, Any]:
    """Load model artifacts from disk.

    Raises FileNotFoundError with a helpful message if artifacts are missing.
    """
    def _p(p):
        return os.path.join(models_dir, p)

    expected = {
        "vectorizer": _p("vectorizer.pkl"),
        "classifier": _p("career_classifier.pkl"),
        "kmeans": _p("kmeans.pkl"),
        "role_skill_map": _p("role_skill_map.pkl"),
    }

    missing = [name for name, path in expected.items() if not os.path.exists(path)]
    if missing:
        # If artifacts are missing, attempt to create them by running train_model.py
        # with the current Python interpreter. This helps deployed apps that don't
        # have prebuilt model artifacts (e.g., Streamlit Cloud) bootstrap themselves.
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "train_model.py"])
        except Exception as e:
            # If creation failed, raise a clear FileNotFoundError including the reason.
            raise FileNotFoundError(
                f"Missing model artifacts: {missing}. Attempted to run train_model.py but failed: {e}"
            )

        # Recompute missing after attempt to create
        missing = [name for name, path in expected.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts: {missing}. Run `python train_model.py` to create them in '{models_dir}'."
            )

    return {k: joblib.load(v) for k, v in expected.items()}


def generate_roadmap(preds: List[str], skills: List[str]) -> List[str]:
    """Return a short 3-step roadmap for the top predicted role.

    preds is expected to be a list like ['Role (score=0.9)', ...]
    """
    if not preds:
        return [
            "Assess your current skills.",
            "Choose a target role and gather learning resources.",
            "Work on projects to demonstrate your skills.",
        ]

    top_role = preds[0].split("(")[0].strip()
    return [
        f"Learn fundamentals and core concepts for {top_role}.",
        "Build 2-3 projects that showcase relevant skills and technologies.",
        "Prepare for interviews: system design, algorithms (if required), and domain questions. Polish your resume and LinkedIn.",
    ]
