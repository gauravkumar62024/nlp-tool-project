import nltk

from tkinter import *
def tokenization():
    data= e.get()

    outlbl.configure(text=nltk.sent_tokenize(data), fg='green')



root=Tk()
root.state('zoomed')
root.configure(background='lightgrey')
title=Label(root,text='ALL NLP TOOLS DEMO',bg='lightgrey',font=('',38,'bold'))
title.place(x=400,y=10)
f=Frame(root)

lbl=Label(root,text='Enter msg:',fg='blue',bg='lightgrey',font=('',20,'bold'))
lbl.pack()


e=Entry(root,font=('',15,'bold'),width="35")
e.pack()

b=Button(root,text='Spam Predict',command=tokenization,font=('',15,'bold'))
b.pack()

outlbl=Label(root,bg='yellow',font=('',20,'bold'))
outlbl.pack()



root.mainloop()