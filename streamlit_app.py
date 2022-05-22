
#Import:

import streamlit as st
import numpy as np
import pytesseract
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import translators as ts
import joblib



#Functions:

# @st.cache
def ocr(img):
  with st.spinner('Loading...'):
    text = pytesseract.image_to_string(img,lang = 'tha+vie+khm+lao+mya+fil')
    from_lang,from_lang1 = predict(text)
    text2 = pytesseract.image_to_string(img,lang = from_lang)
  return text2,from_lang,from_lang1

# @st.cache
def predict(text):
    cv = joblib.load('./cv_model')
    le = joblib.load('./le_model (1)')
    x = cv.transform([text]).toarray() 
    lang = model.predict(x) 
    lang = le.inverse_transform(lang)
    if lang[0] == 'Vietnamese':
      from_lang = 'vie'
    elif lang[0] == 'Thai':
      from_lang = 'tha'
    elif lang[0] == 'Laos':
      from_lang = 'lao'
    elif lang[0] == 'Myanmar':
      from_lang = 'mya'
    elif lang[0] == 'Tagalog':
      from_lang = 'fil'
    elif lang[0] == 'Cambodia':
      from_lang = 'khm'
    return from_lang,lang[0]

# @st.cache
def trans(text, from_lang, to_lang):
    if from_lang == 'vie':
      from_lang2 = 'vi'
    elif from_lang == 'tha':
      from_lang2 = 'th'
    elif from_lang == 'lao':
      from_lang2 = 'lo'
    elif from_lang == 'mya':
      from_lang2 = 'my'
    elif from_lang == 'fil':
      from_lang2 = 'tl'
    elif from_lang == 'khm':
      from_lang2 = 'km'
    if to_lang == 'Vietnamese':
      to_lang2 = 'vi'
    elif to_lang == 'Thai':
      to_lang2 = 'th'
    elif to_lang == 'Laos':
      to_lang2 = 'lo'
    elif to_lang == 'Myanmar':
      to_lang2 = 'my'
    elif to_lang == 'Tagalog':
      to_lang2 = 'tl'
    elif to_lang == 'Cambodia':
      to_lang2 = 'km'
    elif to_lang == 'English':
      to_lang2 = 'en'
    translated = ts.google(text, from_language=from_lang2, to_language=to_lang2)
    return translated

#UI Setup:
CURRENT_THEME = "dark"
IS_DARK_THEME = True
st.set_page_config(page_title='Language Detection', page_icon="👀")
st.set_option('deprecation.showfileUploaderEncoding', False) # disable deprecation error
with open("streamlit_app.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#Customize
st.title("Language Detection 👋")

@st.cache(allow_output_mutation = True)     # enable cache to improve the loading time
def get_model():
    model = joblib.load('./multiNB')
    return model

st.sidebar.title("Group 9")
st.sidebar.markdown("### Group Member")
st.sidebar.markdown("- Chananyu Kamolsuntron")
st.sidebar.markdown("- Intouch Wangtrakoondee")
st.sidebar.markdown("- Nithiwat Pattrapong")
st.sidebar.markdown("- Nguyen Dang")
st.sidebar.markdown("- Thanin Katanyutapant")
st.sidebar.markdown("- Lapin Buranaassavakul")


with st.spinner('Loading Model...'):
    model = get_model()
imgupload = st.file_uploader("Please upload the picture", type=["png", "jpg", "jpeg"])

if imgupload is not None:
  image = Image.open(imgupload)
  st.image(image)
  text,from_lang,from_lang1 = ocr(image)
  st.write("Predicted Language: **{}** 👇".format(from_lang1))
  st.write("**Extracted text:**")
  st.write(text)
  to_lang = st.radio(
    'Language to translate to',
    ('Vietnamese', 'Thai', 'Laos','Tagalog','Cambodia','English'))
  if st.button('Translate'):
     transalate = trans(text,from_lang,to_lang)
     st.write("**Translated:**")
     st.write(transalate)
