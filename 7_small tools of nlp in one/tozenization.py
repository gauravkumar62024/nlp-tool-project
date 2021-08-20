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
lbl.place(x=100,y=200)


e=Entry(root,font=('',15,'bold'),width="35")
e.place(x=350,
		y=205,
		width=400,
		height=100)

b=Button(root,text='Spam Predict',command=mypredict,font=('',15,'bold'))
b.place(x=150,y=450)
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



root.mainloop()