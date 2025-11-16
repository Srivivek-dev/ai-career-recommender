# app.py
import streamlit as st
import pandas as pd
import joblib
from resume_parser import extract_text_from_pdf, parse_skills_from_text
from career_model import CareerModel
from utils import generate_roadmap, load_models

def main():
    st.set_page_config(page_title='AI Career Recommender', layout='wide')
    st.title('AI Career Path Recommender & Resume Analyzer')

    uploaded_file = st.file_uploader('Upload Resume (PDF)', type=['pdf'])
    extra_skills = st.text_input('Add extra skills (comma-separated)', '')

    model_artifacts = load_models()
    career_model = CareerModel(model_artifacts)

    if st.button('Analyze'):
        if not uploaded_file:
            st.error('Please upload a resume PDF.')
        else:
            text = extract_text_from_pdf(uploaded_file)
            skills = parse_skills_from_text(text)

            if extra_skills:
                skills.extend([s.strip().lower() for s in extra_skills.split(',')])

            skills = list(set(skills))

            st.subheader('Extracted Skills')
            st.write(skills)

            preds = career_model.predict_roles_from_skilllist(skills)
            st.subheader('Career Role Predictions')
            st.write(preds)

            score = career_model.skill_score(skills)
            st.metric('Skill Match Score', f'{score:.2f}')

            st.subheader('Missing Skills')
            st.write(career_model.suggest_missing_skills(skills))

            st.subheader('Roadmap')
            for i, step in enumerate(generate_roadmap(preds, skills), 1):
                st.write(f"**Month {i}:** {step}")


if __name__ == '__main__':
    main()
