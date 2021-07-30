import nltk


from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tkinter import *


df=pd.read_csv('dataset/sms.txt',delimiter='\t')
cv=CountVectorizer(stop_words='english')
mnb=MultinomialNB()

def mytrain():
	df.columns=['label','msg']
	X=cv.fit_transform(df.msg).todense()
	y=df.iloc[:,0].values
	mnb.fit(X,y)

def mypredict():
	msg=e.get()
	X_test=cv.transform([msg]).todense()
	pred=mnb.predict(X_test)
	if(pred[0]=='spam'):
		outlbl.configure(text=pred[0],fg='red')
	else:
		outlbl.configure(text=pred[0],fg='green')

mytrain()



z="xyz"
a="pt"
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
    return z,a


# let's begin
def summerize():
    msg1=e.get()
    f = open('page.txt', 'w+')
    f.write(msg1)
    k,y= generate_summary("pri.txt", 2)
    outlbl.configure(text=y, fg='green')
    outlbl.configure(text=k, fg='black')




root=Tk()
root.state('zoomed')
root.configure(background='lightgrey')
title=Label(root,text='ALL NLP TOOLS DEMO',bg='lightgrey',font=('',38,'bold'))
title.place(x=400,y=10)
f=Frame(root)

lbl=Label(root,text='Enter msg:',fg='blue',bg='lightgrey',font=('',20,'bold'))
lbl.place(x=100,y=200)


e=Entry(root,font=('',15,'bold'),width="35")
e.place(x=350,
		y=205,
		width=400,
		height=100)

b=Button(root,text='Spam Predict',command=mypredict,font=('',15,'bold'))
b.place(x=150,y=450)

d=Button(root,text='text summurization',command=summerize,font=('',15,'bold'))
d.place(x=300,y=450)
outlbl=Label(root,bg='yellow',font=('',20,'bold'))
outlbl.place(x=350,y=350)


root.mainloop()