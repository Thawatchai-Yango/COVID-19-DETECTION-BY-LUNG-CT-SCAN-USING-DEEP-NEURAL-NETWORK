# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import sqlite3

from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl

from MAIN_CODE import Ui_MainWindow1 as Ui_HomePage
from PyQt4.QtGui import *
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

class Ui_Dialog(object):
    
    def UserHomeCheck(self):
        self.userHomeWindow=QtGui.QMainWindow()
        self.ui=Ui_HomePage()
        self.ui.setupUii(self.userHomeWindow)
        self.userHomeWindow.show()

           
    def loginCheck(self):
        username1=self.uname_lineEdit.text()
        username =str(username1)
        password1=self.pass_lineEdit.text()
        password = str(password1)
        print("password is:"+password)
        connection=sqlite3.connect("multiD.db")
        s="select *from userdetails where username='"+username+"' and password='"+password+"'"
        print("query is:"+s)
        result=connection.execute(s)
        if(len(result.fetchall())>0):
         print("user found!")
         self.UserHomeCheck()

        else:
         print("user not fount!")
         self.showmsg()

        cursor = connection.cursor()
        cursor.execute("select *from userdetails where username='"+username+"' and password='"+password+"'")
        result = cursor.fetchall();
        for row in result:
            print("User Name      : ", row[0])
            print("User Email-id  : ", row[1])
            print("User Mobile-no : ", row[2])
            print("\n")

            name    = ("User Name      : ", row[0])
            email   = ("User Email-id  : ", row[1])
            mob_no  = ("User Mobile-no : ", row[2])
            
        self.print_personal_detail( user_name  = name , user_email = email,user_mob   = mob_no)
        connection.commit()
        connection.close()
    
    def signupCheck(self):
        self.signUpShow()
        print("Signup button clicked !")

    def showmsg(self):
        self.showdialog1()

    def showdialog1(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Login failed")
        msg.setInformativeText("Please enter correct details ")
        msg.setWindowTitle("Status")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        retval = msg.exec_()
        print
        "value of pressed message box button:", retval

    def setinUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        #Dialog.resize(1200, 800)
        Dialog.setFixedSize(1100,690)
        Dialog.setStyleSheet(_fromUtf8("background-image: url(login_bg.jpg);"))
        
        self.u_user_label = QtGui.QLabel(Dialog)
        self.u_user_label.setGeometry(QtCore.QRect(530 , 250, 130, 30))
        self.u_user_label.setObjectName(_fromUtf8("u_user_label"))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.u_user_label.setFont(font)
           
        self.pass_label = QtGui.QLabel(Dialog)
        self.pass_label.setGeometry(QtCore.QRect(530, 300, 130, 30))
        self.pass_label.setObjectName(_fromUtf8("pass_label"))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pass_label.setFont(font)
       
        
        self.uname_lineEdit = QtGui.QLineEdit(Dialog)
        self.uname_lineEdit.setGeometry(QtCore.QRect(700, 250, 300, 30))
        self.uname_lineEdit.setText(_fromUtf8(""))
        self.uname_lineEdit.setObjectName(_fromUtf8("uname_lineEdit"))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.uname_lineEdit.setFont(font)
       
        
        self.pass_lineEdit = QtGui.QLineEdit(Dialog)
        self.pass_lineEdit.setGeometry(QtCore.QRect(700, 300, 300, 30))
        self.pass_lineEdit.setObjectName(_fromUtf8("pass_lineEdit"))
        self.pass_lineEdit.setEchoMode(QtGui.QLineEdit.Password)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.pass_lineEdit.setFont(font)
        
        
        self.login_btn = QtGui.QPushButton(Dialog)
        self.login_btn.setGeometry(QtCore.QRect(700, 350, 300, 30))
        self.login_btn.setObjectName(_fromUtf8("login_btn"))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.login_btn.setFont(font)
        #####################Button Event####################################
        self.login_btn.clicked.connect(self.loginCheck)
        ####################################################################

        self.exit = QtGui.QPushButton(Dialog)
        self.exit.setGeometry(QtCore.QRect(700, 400, 300, 30))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.exit.setFont(font)
        self.exit.setObjectName(_fromUtf8("exit"))
        #self.exit.setStyleSheet("background-color: black")
        #########################EVENT##############
        self.exit.clicked.connect(self.closeall)


        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def closeall(self):
        quit()

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "LOGIN FORM", None))
        self.u_user_label.setText(_translate("Dialog", "NAME", None))
        self.pass_label.setText(_translate("Dialog", "PASSWORD", None))
        self.login_btn.setText(_translate("Dialog", "LOGIN TO SYSTEM", None))
        self.exit.setText(_translate("Dialog", "CLICK TO EXIT", None))

    def print_personal_detail(self,user_name,user_email,user_mob):
        print_pdf_name  = user_name
        print_pdf_email = user_email
        print_pdf_mob   = user_mob
        outfile = "template.pdf"
        template = PdfReader("template_login.pdf", decompress=False).pages[0]
        template_obj = pagexobj(template)
        canvas = Canvas(outfile)
        xobj_name = makerl(canvas, template_obj)
        canvas.doForm(xobj_name)
        canvas.setFont("Helvetica", 16)
        canvas.drawString(250, 500, str(print_pdf_name))
        canvas.drawString(250, 450, str(print_pdf_email))
        canvas.drawString(250, 400, str(print_pdf_mob))
        canvas.save()
                
    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()

     


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setinUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

