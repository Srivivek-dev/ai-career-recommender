# Practical Resume Parser

This project parses resumes and trains a career prediction model.

Required Python packages are listed in `requirements.txt`.

Quick setup

1. Create or activate a Python environment (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Download spaCy English model:

   python -m spacy download en_core_web_sm

4. Verify imports (optional):

   python -c "import PyPDF2, spacy; nlp = spacy.load('en_core_web_sm'); print(PyPDF2.__version__, spacy.__version__)"

Notes

- If your editor (e.g., VS Code with Pylance) still shows missing import warnings, ensure the editor is using the same Python interpreter as your environment.
- macOS users may need to install Xcode command line tools for some packages.