import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from tkinter import *
import tkinter as tk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize



z="xyz"
a="pt"

df = pd.read_csv('dataset/sms.txt', delimiter='\t')
cv = CountVectorizer(stop_words='english')
mnb = MultinomialNB()


def mytrain():
    df.columns = ['label', 'msg']
    X = cv.fit_transform(df.msg).todense()
    y = df.iloc[:, 0].values
    mnb.fit(X, y)


def mypredict():
    msg = e.get('1.0', tk.END)
    X_test = cv.transform([msg]).todense()
    pred = mnb.predict(X_test)
    if (pred[0] == 'spam'):
        outlbl.insert(tk.END, pred[0])
    else:
        outlbl.insert(tk.END, pred[0])


mytrain()
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    global a
    a=ranked_sentence
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    global z
    z = summarize_text
    print("Indexes of top ranked_sentence order are ", ranked_sentence,"\n","Summarize Text: \n", ". ".join(summarize_text))
    return z


# let's begin
def summerize():
    msg1=e.get('1.0',tk.END)
    f = open('page.txt', 'w+')
    f.write(msg1)
    k= generate_summary("pri.txt", 2)
    k=''.join(k)
    outlbl.insert(tk.END,k)

def clear_text():
    outlbl.delete('1.0',END)
def sen_token():
    msg2= e.get('1.0', tk.END)
    sen_tok=nltk.sent_tokenize(msg2)
    outlbl.insert(tk.END,sen_tok)

def word_token():
    msg3= e.get('1.0', tk.END)
    word_tok=nltk.word_tokenize(msg3)
    delim = "/ "

    res = ''

    # using loop to add string followed by delim
    for ele in word_tok:
        res = res + str(ele) + delim

    # printing result
    outlbl.insert(tk.END,res)

def pos():
    msg4 = e.get('1.0', tk.END)
    pos = nltk.pos_tag(nltk.word_tokenize(msg4))
    outlbl.insert(tk.END, pos)


def lementization():
    msg4 = e.get('1.0', tk.END)
    import nltk


    lemmatizer = WordNetLemmatizer()

    # Define function to lemmatize each word with its POS tag

    # POS_TAGGER_FUNCTION : TYPE 1
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    sentence = msg4
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    outlbl.insert(tk.END,lemmatized_sentence)

def ner():
    sent = e.get('1.0', tk.END)
    words = nltk.word_tokenize(sent)
    pos_tag = nltk.pos_tag(words)
    namedEntity = nltk.ne_chunk(pos_tag)
    #namedEntity.draw()
    outlbl.insert(tk.END,namedEntity.draw())

def parser():
    grammar = nltk.CFG.fromstring("""
      S -> NP VP
      VP -> V NP | V NP PP
      PP -> P NP
      V -> "saw" | "slept" | "walked"
      NP -> "Rahul" | "Anjali" | Det N | Det N PP
      Det -> "a" | "an" | "the" | "my"
      N -> "man" | "dog" | "cat" | "telescope" | "park"
      P -> "in" | "on" | "by" | "with"
      """)
    sent1 = e.get('1.0', tk.END)
    sent = sent1.split()
    parser = nltk.RecursiveDescentParser(grammar)
    for tree in parser.parse(sent):
        print(tree)
        #tree.draw()
    outlbl.insert(tk.END, tree.draw())
def ngram():
    sent1 = e.get('1.0', tk.END)
    string = [sent1]
    vect1 = CountVectorizer()
    vect1.fit_transform(string)
    d= vect1.get_feature_names()
    delim = ", "

    res = ''

    # using loop to add string followed by delim
    for ele in d:
        res = res + str(ele) + delim

    # printing result
    outlbl.insert(tk.END, res)




#Creating tkinter main window
root = tk.Tk()
root.geometry('1100x600')
root.title("NLP TOOLS")

# Title Label
lbl=Label(root,text='Enter msg:',fg='blue',bg='lightgrey',font=('',20,'bold'))
lbl.place(x=100,y=0)
lbl2=Label(root,text='output:',fg='blue',bg='lightgrey',font=('',20,'bold'))
lbl2.place(x=600,y=0)
# Creating scrolled text
# area widget
e = scrolledtext.ScrolledText(root,
                                      wrap=tk.WORD,
                                      width=400,
                                      height=200,
                                      font=("Times New Roman",
                                            15))

e.place(x=0,y=50,width=400, height=200)

b=Button(root,text='Text Summarize',command=summerize,font=('',15,'bold'))
b.place(x=350,y=400)
d=Button(root,text='Predict',command=mypredict,font=('',15,'bold'))
d.place(x=260,y=400)
cl=Button(root,text='Clear',command=clear_text,font=('',15,'bold'))
cl.place(x=400,y=500)
sent_t=Button(root,text='Sentence Tokenization',command=sen_token,font=('',15,'bold'))
sent_t.place(x=250,y=300)

word_t=Button(root,text='Word Tokens',command=word_token,font=('',15,'bold'))
word_t.place(x=100,y=300)

pos=Button(root,text='Pos Tagging',command=pos,font=('',15,'bold'))
pos.place(x=500,y=300)

l1=Button(root,text='Lementization',command=lementization,font=('',15,'bold'))
l1.place(x=100,y=400)

ner=Button(root,text='NER',command=ner,font=('',15,'bold'))
ner.place(x=540,y=400)

per=Button(root,text='Parsing',command=parser,font=('',15,'bold'))
per.place(x=100,y=500)

ngrm = Button(root, text='Bags of word', command=ngram, font=('', 15, 'bold'))

ngrm.place(x=200, y=500)
outlbl=scrolledtext.ScrolledText(root,
                                      wrap=tk.WORD,
                                      width=400,
                                      height=200,
                                      font=("Times New Roman",
                                            15))
outlbl.place(x=500,y=50,width=400, height=200)
# Placing cursor in the text area
e.focus()
root.mainloop()