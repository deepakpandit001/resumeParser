import nltk
import docx2txt
import locationtagger
from nltk.corpus import stopwords
import re
import spacy
import json
import PyPDF2
from pathlib import Path
import requests

from spacy.matcher import Matcher
# load pre-trained model
nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(stopwords.words('english'))
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)


def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


def get_phone_numbers(string):
    r = re.compile(
        r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]


def extract_name(resume_text):
    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern], on_match=None)

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text


EDUCATION = [
    'BE', 'B.E.', 'B.E', 'BS', 'B.S',
    'ME', 'M.E', 'M.E.', 'MS', 'M.S',
    'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
    'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
]


def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    skills = ["machine learning",
              "deep learning",
              "nlp",
              "natural language processing",
              "mysql",
              "sql",
              "django",
              "computer vision",
              "tensorflow",
              "opencv",
              "mongodb",
              "artificial intelligence",
              "ai",
              "flask",
              "robotics",
              "data structures",
              "python",
              "c++",
              "matlab",
              "css",
              "html",
              "github",
              "php"]

    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


# initializing sample text
sample_text = "India has very rich and vivid culture\
	widely spread from Kerala to Nagaland to Haryana to Maharashtra. " \
        "Delhi being capital with Mumbai financial capital.\
	Can be said better than some western cities such as " \
        " Munich, London etc. Pakistan and Bangladesh share its borders"

# extracting entities.


def get_Address(str):
    dictAdd = {'Resume': 'Address'}
    place_entity = locationtagger.find_locations(text=str)
    # getting all countries
    dictAdd['countries'] = place_entity.countries
    dictAdd['regions'] = place_entity.regions
    # getting all cities
    dictAdd['cities'] = place_entity.cities
    return dictAdd


# ------------
# example_09.py

# if These file is not downloaded

'''nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')'''


RESERVED_WORDS = [
    'school',
    'college',
    'univers',
    'academy',
    'faculty',
    'institute',
    'faculdades',
    'Schola',
    'schule',
    'lise',
    'lyceum',
    'lycee',
    'polytechnic',
    'kolej',
    'ünivers',
    'okul',
]


def extract_education(input_text):
    organizations = []

    # first get all the organization names using nltk
    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))

    # we search for each bigram and trigram for reserved words
    # (college, university etc...)
    education = set()
    for org in organizations:
        for word in RESERVED_WORDS:
            if org.lower().find(word):

                education.add(org)

    return education


def pdfToJson(str):
    pdfFileObj = open(str, 'rb')
    #pdfFileObj = open('ShaliniRsm.pdf', 'rb')
    #pdfFileObj = open('myRsm.pdf', 'rb')
    #pdfFileObj = open('SachinRsm.pdf', 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    print(pdfReader.numPages)
    pageObj = pdfReader.getPage(0)
    value = pageObj.extractText()
    pdfFileObj.close()
    return textToJson(value)


def pdftToText(str):
    pdfFileObj = open(str, 'rb')
    #pdfFileObj = open('ShaliniRsm.pdf', 'rb')
    #pdfFileObj = open('myRsm.pdf', 'rb')
    #pdfFileObj = open('SachinRsm.pdf', 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    print(pdfReader.numPages)
    pageObj = pdfReader.getPage(0)
    value = pageObj.extractText()
    pdfFileObj.close()
    return value


def textToJson(value):
    dict = {'Resume': 'Detail'}
    name = extract_namesJ(value)
    email = get_email_addresses(value)
    skill = extract_skillsJ(value)
    phone = get_phone_numbers(value)
    education = extract_education(value)
    dict['Name'] = name
    dict['Email'] = email
    dict['Phone'] = phone
    dict['Education'] = email
    dict['Skill'] = skill
    josnAddress = get_Address(value)
    dict['Address'] = josnAddress
    return dict


def getPdfFromUrl(str):
    filename = Path('resumeParse.pdf')
    response = requests.get(str)
    filename.write_bytes(response.content)

#############################################


def extract_namesJ(resume_text):
    nlp_text = nlp(resume_text)
    name_file = open('names.json')
    name_data = json.load(name_file)
    user_name = name_data["names"]
    names_lower = [name.lower() for name in user_name]
    name_file.close()

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


def extract_skillsJ(resume_text):
    nlp_text = nlp(resume_text)
    skills_file = open('skills.json')
    skills_data = json.load(skills_file)
    skills = skills_data["skills"]
    skills_lower = [name.lower() for name in skills]
    skills_file.close()
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
