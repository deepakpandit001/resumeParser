import json
import nltk
import docx2txt
import locationtagger
from nltk.corpus import stopwords
import re
import spacy
import PyPDF2
from pathlib import Path
import requests

from spacy.matcher import Matcher
# load pre-trained model
nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(stopwords.words('english'))
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

import PyPDF2
pdfFileObj = open('ppp.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)
pageObj = pdfReader.getPage(0)
pdfText = pageObj.extractText()

def extract_skills(resume_text,skills):
    nlp_text = nlp(resume_text)
    skills_lower = [name.lower() for name in skills]
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills_lower:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills_lower:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def extract_names(resume_text,names):
    nlp_text = nlp(resume_text)
    names_lower = [name.lower() for name in names]

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in names_lower:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in names_lower:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]





f = open('skills.json')
data = json.load(f)
name_file = open('names.json')
name_data = json.load(name_file)
user_name = name_data["names"]
arr = data["skills"]
haveSkill = extract_skills(pdfText,arr)
haveName = extract_names(pdfText,user_name)
print(haveSkill)
print(haveName)
















pdfFileObj.close()
