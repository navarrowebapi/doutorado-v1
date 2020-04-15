import re
import numpy as np
import pandas as pd
from pprint import pprint

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

print("OK1")

# NLTK Stop words
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('portuguese')
#stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

print("OK2")

#CORPUS
import PyPDF2
import re
import glob

#your full path of directory
mypath = "/PLN/corpus-artigos"
for file in glob.glob(mypath + "/*.pdf"):
    print(file)
    if file.endswith('.pdf'):
        fileReader = PyPDF2.PdfFileReader(open(file, "rb"))
        count = 0
        count = fileReader.numPages
        while count >= 0:
            count -= 1
            pageObj = fileReader.getPage(count)
            text = pageObj.extractText()
            print(text)
        #num = re.findall(r'[0-9]+', text)
        #print(num)
    else:
        print("not in format")

print("OK3")

