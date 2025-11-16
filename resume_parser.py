# resume_parser.py
import PyPDF2
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

COMMON_SKILLS = [
    'python','java','c++','c','c#','javascript','sql','html','css','react','node',
    'tensorflow','pytorch','machine learning','deep learning','nlp','computer vision',
    'docker','kubernetes','aws','azure','gcp','git','pandas','numpy','scikit-learn'
]

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_skills_from_text(text):
    text_low = text.lower()
    found = set()

    for s in COMMON_SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text_low):
            found.add(s)

    if nlp:
        doc = nlp(text)
        for token in doc:
            if token.text.lower() in COMMON_SKILLS:
                found.add(token.text.lower())

    return sorted(found)
