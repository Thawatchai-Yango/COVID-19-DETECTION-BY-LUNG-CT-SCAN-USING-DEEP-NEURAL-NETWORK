# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UploadVideo.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!
import tkinter
import tkinter.filedialog
from tkinter import messagebox
from tkinter import *
import tkinter as tk

from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl

import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
#from Sim_SV import Calc_Wt
import time
from tkinter import filedialog
import tkinter.messagebox
import cv2
from PyQt4 import QtCore, QtGui 
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

##from sklearn import svm, datasets
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import confusion_matrix

import tensorflow as tf, sys
import cv2
import time
from tkinter import filedialog
import tkinter.messagebox
import numpy as np
import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
root=tkinter.Tk()
root.wm_withdraw()

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('TRUE CLASS')
    plt.xlabel('PREDICTED CLASS')
    plt.tight_layout()
    
def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default
    

print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        #MainWindow.resize(850,450)
        MainWindow.setFixedSize(1100,690)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(main_bg.jpg);\n"""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
#################################################################
        '''self.topTitle = QtGui.QPushButton(self.centralwidget)
        self.topTitle.setGeometry(QtCore.QRect(130, 50, 500, 80))
        self.topTitle.setStyleSheet("background-color: blue")
        font = QtGui.QFont()
        font.setPointSize(30)
        self.topTitle.setFont(font)'''
#################################################################
       
        
#################################################################
        
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(555, 250, 400, 70))
        self.pushButton_2.clicked.connect(self.show1)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);\n"
                                                  "color: rgb(0, 0, 204);"))
        #self.pushButton_2.setStyleSheet(_fromUtf8("background-color: blue"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
       
#################################################################
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(555, 350, 400, 70))
        self.pushButton_4.clicked.connect(self.show2)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);\n"
                                                "color: rgb(0, 0, 204);"))
        #self.pushButton_4.setStyleSheet(_fromUtf8("background-color: blue"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
#################################################################      
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(555, 450, 400, 70))
        self.pushButton.clicked.connect(self.quit)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);\n"
                                                "color: rgb(0, 0, 204);"))
        #self.pushButton.setStyleSheet(_fromUtf8("background-color: blue"))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################
        self.result = QtGui.QPushButton(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(555, 550, 130, 80))
        self.result.setStyleSheet(_fromUtf8("background-color: rgb(0, 0,204);\n"
                                            "color: rgb(0, 0,204);"))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.result.setFont(font)
        
        self.output_rd = QtGui.QTextBrowser(self.centralwidget)
        self.output_rd.setGeometry(QtCore.QRect(700, 550, 255, 80))
        self.output_rd.setStyleSheet(_fromUtf8("background-color: rgb(255, 0,0);\n"
                                                "color: rgb(255, 0,0);"))
        self.output_rd.setObjectName("output_rd")
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.output_rd.setFont(font)

#################################################################
        self.result_accu = QtGui.QPushButton(self.centralwidget)
        self.result_accu.setGeometry(QtCore.QRect(555, 630, 130, 40))
        self.result_accu.setStyleSheet(_fromUtf8("background-color: rgb(0, 0,204);\n"
                                            "color: rgb(0, 0,204);"))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.result_accu.setFont(font)

        self.output_accu = QtGui.QTextBrowser(self.centralwidget)
        self.output_accu.setGeometry(QtCore.QRect(700, 630, 255, 40))
        self.output_accu.setStyleSheet(_fromUtf8("background-color: rgb(255, 0,0);\n"
                                                "color: rgb(255, 0,0);"))
        self.output_accu.setObjectName("output_accu")
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.output_accu.setFont(font)

        
#################################################################
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "ADMIN PANEL FOR COVID CT-SCAN TESTING", None))
        
        #self.topTitle.setText(_translate("MainWindow", "COVID CT-SCAN TESTING ", None))
        self.pushButton_2.setText(_translate("MainWindow", "INPUT CT-SCAN IMAGE ", None))
        self.pushButton_4.setText(_translate("MainWindow", "ACCURACY", None))
        #self.pushButton_5.setText(_translate("MainWindow", "ACCURACY", None))
        self.pushButton.setText(_translate("MainWindow", "EXIT THE PROGRAM", None))
        self.result.setText(_translate("MainWindow", "RESULT", None))
        self.result_accu.setText(_translate("MainWindow", "ACCURACY(%)", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()

    def print_covid(self,test_text,accu_text):
        print_pdf_test = test_text
        print_pdf_accu = accu_text
        outfile = "Covid_Test.pdf"
        template = PdfReader("template.pdf", decompress=False).pages[0]
        template_obj = pagexobj(template)
        canvas = Canvas(outfile)
        xobj_name = makerl(canvas, template_obj)
        canvas.doForm(xobj_name)
        canvas.setFont("Helvetica", 16)
        canvas.drawString(250, 200, str(print_pdf_test))
        canvas.drawString(250, 150, "Accuracy : ")
        canvas.drawString(350, 150, str(print_pdf_accu))
        canvas.save()
        
    def show1(self):
        self.output_rd.setText("")
        self.output_accu.setText("")
        image_path= filedialog.askopenfilename(filetypes = (("BROWSE CT IMAGE", "*.jpg"), ("All files", "*")))
        I=cv2.imread(image_path)
        #I=cv2.imread('TESTFULL.jpg')
        cv2.imshow('INPUT IMAGE',I);
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # RESIZING
        img = cv2.resize(I,(512,512),3)
        cv2.imshow('RESIZED IMAGE',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # MEDIAN FILTERED
        img1 = cv2.medianBlur(img,5)
        cv2.imshow('MEDIAN IMAGE',img1)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # GRAY CONVERSION
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        cv2.imshow('GRAY IMAGE',gray)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # Open label file
        label_lines = [line.rstrip() for line
            in tf.gfile.GFile("retrained_labels.txt")]
        # CNN trained file
        with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')



        # TEST given input image
        with tf.Session() as sess:
             softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
             print(softmax_tensor)
             predictions = sess.run(softmax_tensor, 
             {'DecodeJpeg/contents:0': image_data})
             # Confidenece
             top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
             human_string = label_lines[0]
             score1 = predictions[0][0]
             print('PREDICTION----:\n',predictions)
             #print('SCORES--:\n',score1)
             CurID=np.argmax(predictions)
             print('PREDICTED CLASS INDEX\n',CurID)

        print('-----------------------------------------------------\n')
        print('----------------------RESULT-------------------------\n')

        #window = Tk()
        #window.title('RESULT OF PREDICTION')
        #window.geometry("700x200")

            
        if np.max(predictions)>=0.4:
            
            if CurID==0:
                print('CLASS: NORMAL--:\n')
                #win = Message(window, text="CLASS: NORMAL--:\n")
                #win.config(bg='black', fg ='white', font=('Times New Roman', '30'))
                #win.pack(fill = BOTH, expand = True)'''
                self.output_rd.setText("CLASS: NORMAL--:")
                accuracy = np.max(predictions)*100.00
                self.output_accu.setText(str(accuracy))
                self.print_covid(test_text = "CLASS: NORMAL ", accu_text = accuracy)
                
            if CurID==1:
                print('CLASS: COVID: I STAGE--:\n')
                #win = Message(window, text="CLASS: COVID: I STAGE--:\n")
                #win.config(bg='black', fg ='white', font=('Times New Roman', '30'))
                #win.pack(fill = BOTH, expand = True)
                self.output_rd.setText("CLASS: COVID: I STAGE--:")
                accuracy = np.max(predictions)*100.00
                self.output_accu.setText(str(accuracy))
                self.print_covid(test_text = "CLASS: COVID: I STAGE ", accu_text = accuracy) 
                
            if CurID==2:
                print('CLASS: COVID: II STAGE--:\n')
                #win = Message(window, text="CLASS: COVID: II STAGE--:\n")
                #win.config(bg='black', fg ='white', font=('Times New Roman', '30'))
                #win.pack(fill = BOTH, expand = True)
                self.output_rd.setText("CLASS: COVID: II STAGE--:")
                accuracy = np.max(predictions)*100.00
                self.output_accu.setText(str(accuracy))
                self.print_covid(test_text = "CLASS: COVID: II STAGE ", accu_text = accuracy) 
                
            if CurID==3:
                print('CLASS: COVID: SEVER STAGE--:\n')
                #win = Message(window, text="CLASS: COVID: SEVER STAGE--:\n")
                #win.config(bg='black', fg ='white', font=('Times New Roman', '30'))
                #win.pack(fill = BOTH, expand = True)
                self.output_rd.setText("CLASS: COVID: SEVERE STAGE--:")
                accuracy = np.max(predictions)*100.00
                self.output_accu.setText(str(accuracy))
                self.print_covid(test_text = "CLASS: COVID: SEVERE STAGE ", accu_text = accuracy)

                
        else:
            print('UNABLE TO PREDICT--:\n')
            #win = Message(window, text="UNABLE TO PREDICT--:\n")
            #win.config(bg='black', fg ='white', font=('Times New Roman', '30'))
            #win.pack(fill = BOTH, expand = True)
            self.output_rd.setText("UNABLE TO PREDICT--:")
            accuracy = np.max(predictions)*100.00
            self.output_accu.setText(str(accuracy))

            
        
    def show2(self):
        import socket
        import time
        import cv2
        import os
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras import regularizers
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.optimizers import Adam, SGD
        import pickle

        # define parameters
        CLASS_NUM = 5
        BATCH_SIZE = 16
        EPOCH_STEPS = int(4323/BATCH_SIZE)
        IMAGE_SHAPE = (224, 224, 3)
        IMAGE_TRAIN = 'TRNMDL'
        MODEL_NAME = 'GNET.h5'
        def inception(x, filters):
                path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
                path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)
                path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
                path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)
                return Concatenate(axis=-1)([path1,path2,path3,path4])
        def auxiliary(x, name=None):
                layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
                layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Flatten()(layer)
                layer = Dense(units=256, activation='relu')(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
                return layer

        layer_in = Input(shape=IMAGE_SHAPE)
        layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
        layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
        layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
        aux1  = auxiliary(layer, name='aux1')
        layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
        layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
        layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
        aux2  = auxiliary(layer, name='aux2')
        layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
        layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
        layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=256, activation='linear')(layer)
        main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
        model = Model(inputs=layer_in, outputs=[main, aux1, aux2])

        print(model.summary())
        file= open("TRNMDL.obj",'rb')
        cnf_matrix = pickle.load(file)
        file.close()

        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:3,0:3], classes=['Normal ','Mild','Sever'], normalize=True,title='Proposed Method')
        plt.show()


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())
    

