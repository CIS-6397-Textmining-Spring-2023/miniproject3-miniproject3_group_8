import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import bigrams
from nltk.probability import FreqDist
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# from wordcloud import STOPWORDS, ImageColorGenerator

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data,extraWords=None):
    stop_words = stopwords.words('english')
    
    # If extra stopwords are provided, add them to the stopword list
    if extraWords is not None:
        for stop_word in extraWords:
            stop_words.append(stop_word)
    
    # Remove the stopwrods from the tokenized data
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data,stem=False,removeStop=False,extraWords=None):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    if removeStop:
        data = remove_stop_words(data,extraWords)
    data = convert_numbers(data)
    if stem:
        data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    if stem:
        data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    if removeStop:
        data = remove_stop_words(data,extraWords) #needed again as num2word is giving stop words 101 - one hundred and one
    return data
# Used to display how far along an iterative process is

def print_progress(curr,end,len=20):
    frac=curr/end
    lenDashes=int(frac*len)
    if curr==end:
        ending='\n'
    else:
        ending='\r'
    strOut='['+lenDashes*'='+'>'+(len-lenDashes)*' '+']'+' '+str(int(frac*100))+'%'
    print(strOut,end=ending)

def readCorpus(corpus):
    data_dir=corpus
    texts=[]
    
    for file in os.listdir(data_dir):
        # Update progress bar
        print_progress(os.listdir(data_dir).index(file),len(os.listdir(data_dir))-1)
        
        # Read text
        t=open(data_dir+"/"+file,encoding="utf8",errors="ignore")
        temp_txt=t.read().strip()
        t.close()
        texts.append(temp_txt)

    return texts

def preprocessText(texts,stem=False,removeStop=False,extraWords=None):
    data=[]
    tokens=[]
    for i in range(len(texts)):
        # Update loading bar
        print_progress(i,len(texts)-1)
        
        # Preprocess the ith text, giving options for stemming and stopword removal
        temp_txt=preprocess(texts[i],stem,removeStop,extraWords)        
        data.append(temp_txt)
        tokens.append(word_tokenize(str(temp_txt)))
    return data,tokens
# Get the k most frequently used tokens from a text

def getkMostUsed(data,k):
    freqDist=Counter(str(data).split()).most_common()
    return freqDist[:k]
# Get the k most frequently used bigrams from tokenized text

def topkBigrams(tokens,k):
    allTokens=[]
    # Collect all tokens into a single list
    for tokensList in tokens:
        for token in tokensList:
            allTokens.append(token)
    # Get the bigrams
    bgrams = bigrams(allTokens)
    #compute frequency distribution for all the bigrams in the text
    fdist = FreqDist(bgrams)
    return fdist.most_common(k)
# Plot the top k words or bigrams using a sorted dictionary whose keys are tokens or bigrams and whose values are the number of times the key appears

def plotTopk(tokenDict,k,x_label='Word',y_label='Occurances',my_title='',xFS=None,fwidth=7,fheight=5):
    xvals=[]
    yvals=[]
    count=0

    # Get all the keys and values
    for item in tokenDict:
        xvals.append(str(remove_apostrophe(str(remove_punctuation(str(item[0]))))))
        yvals.append(item[1])
        count+=1
        if count==k:
            break

    plt.figure(figsize=(fwidth,fheight))
    # Change xFS(x Font size) - the fontsize of the x axis - if desired
    if xFS is not None:
        plt.tick_params(axis='x', labelsize=xFS)

    # Plot bar graph
    plt.bar(xvals, yvals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(my_title)

def genWordCloud(data,max_words=100,bigrams=False):
    text=''
    if bigrams:
        for item in data:
            temp=remove_apostrophe(str(item[0]))
            temp=remove_punctuation(temp)
            temp=str(temp)[1:-1].replace(' ','_')
            text+=temp+' '
    
    else:
        for doc in data:
            text+=str(doc)+' '

    wordcloud=WordCloud(max_font_size=50, max_words=max_words, background_color="white").generate(text)
    return wordcloud