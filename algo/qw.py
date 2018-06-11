import multiprocessing
import os
import tkinter as tk
from tkinter import *

def call_ml():
    os.system('python3 ml.py')
def call_mobile_ml():
    os.system('python3 ml_mobile.py')
def play_video():
    os.system('python3 play.py')
def show_graph():
    os.system('python3 graph.py')

if __name__ == "__main__":
    # -------configure window
    root = tk.Tk()
    var1 = StringVar()

    label1 = Message( root, textvariable=var1, relief=RAISED,aspect=150,width=2000,pady=50,padx=800,bg='light sea green',font=("Helvetica", 30, "bold") )

    var1.set("INT-Vision")
    label1.pack()
    root.geometry("%dx%d+%d+0" % (1000, 1000,500))

    startbutton=tk.Button(root,width=15,height=2,text='WEBCAM START',command=call_ml)
    startbutton.place(x=100,y=100)
    #stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
    startbutton.pack()
    #stopbutton.pack()
    bu=tk.Button(root,width=15,height=2,text='MOBILE CAM START',command=call_mobile_ml)
    bu.place(x=100,y=100)
    #stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
    bu.pack()
    bu1=tk.Button(root,width=15,height=2,text='PLAY VIDEO',command=play_video)
    bu1.place(x=100,y=100)
    #stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
    bu1.pack()
    bu2=tk.Button(root,width=15,height=2,text='SHOW GRAPH',command=show_graph)
    bu2.place(x=100,y=100)
    #stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
    bu2.pack()


    var2=StringVar()
    label2 = Message( root, textvariable=var2, relief=RAISED,aspect=150,width=2000,pady=50,padx=800,bg='light sea green',font=("Helvetica", 30, "bold"), )
    var2.set("Red shows Disengaged and Blue shows Engaged")
    label2.pack(side=BOTTOM)


    # -------begin
    root.mainloop()
