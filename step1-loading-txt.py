import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.corpora import Dictionary
import gensim

#--------------Leitura dos TXTs e criação do CORPUS------------------
docs_dir = "/PLN/corpus-artigos"

# important to indicate there is another extension before the txt one with '.*.txt'
documents = PlaintextCorpusReader(docs_dir, '.*.txt', encoding='latin-1') # para todos os artigos no diretório acima
#documents = PlaintextCorpusReader(docs_dir, 'artigo1.txt', encoding='latin-1') # para 1 artigo somente
corpus = []

for fileid in documents.fileids():
    document = documents.raw(fileids = fileid)
    corpus.append(document)

print("OK1")
#-------------------------------------------------------------------

tokenizer = RegexpTokenizer(r'\w+')
# Stop words em português
from nltk.corpus import stopwords
stop_words = stopwords.words('portuguese')
_new_stopwords_to_add = ['1','2','3','4','5','6','7','8','9','0', 'ser']
stop_words.extend(_new_stopwords_to_add)
print("OK2")
#-----------------

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

# loop through document list
for i in corpus:
   
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stop_words]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus2 = [dictionary.doc2bow(text) for text in texts]
# View
#print(corpus2[:1])

# Human readable format of corpus (term-frequency)
#print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus2[:1]])

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus2,
                                           id2word=dictionary,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Visualize the topics
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(lda_model, corpus2, dictionary)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')

print("COMPLETE")




