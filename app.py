from flask import Flask, request, render_template, send_from_directory,session,flash
import pandas as pd
import string
import os
import smtplib
import mysql.connector
import numpy as np
from datetime import timedelta
import sys
from PIL import Image
import base64
import io
import re

import PIL.Image
from datetime import datetime
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the random string'
classes = ['System predicted disease as BrownSpot',
           'System predicted disease as Healthy',
           'System predicted disease as LeafBlast',
           'System predicted disease as LeafBlight']


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/reg")
def reg():
    return render_template("ureg.html")
@app.route('/regback',methods = ["POST"])
def regback():
    if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        pwd=request.form['pwd']
        pno=request.form['pno']



    #email = request.form["email"]


        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="rice_leaf"
        )
        mycursor = mydb.cursor()
        sql = "select * from ureg"
        result = pd.read_sql_query(sql, mydb)
        email1 = result['email'].values
        print(email1)
        if email in email1:
            flash("email already existed","warning")
            return render_template('ureg.html', msg="email existed")
        sql = "INSERT INTO ureg (name,email,pwd,pno) VALUES(%s,%s,%s,%s)"
        val = (name, email, pwd, pno)
        mycursor.execute(sql, val)
        mydb.commit()
        flash("Your registration successfully completed", "success")

    return render_template('user.html', msg="registered successfully")
    print("Successfully Registered")

@app.route('/userlog',methods=['POST', 'GET'])
def userlog():
    global name, name1
    global user
    if request.method == "POST":

        username = request.form['email']
        password1 = request.form['pwd']
        print('p')
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="rice_leaf")
        cursor = mydb.cursor()
        sql = "select * from ureg where email='%s' and pwd='%s'" % (username, password1)
        print('q')
        x = cursor.execute(sql)
        print(x)
        results = cursor.fetchall()
        print(results)
        if len(results) > 0:
            print('r')
            # session['user'] = username
            # session['id'] = results[0][0]
            # print(id)
            # print(session['id'])
            flash("Welcome to website", "success")
            return render_template('userhome.html', msg=results[0][1])
        else:
            flash("Invalid Email/password", "danger")
            return render_template('user.html', msg="Login Failure!!!")

    return render_template('user.html')
@app.route("/userhome")
def userhome():
    return render_template("userhome.html")

@app.route("/upload", methods=["POST","GET"])
def upload():
    print('a')
    if request.method=='POST':

        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('D:/RICE LEAF DISEASE DETECTION/imges/', fn)
        myfile.save(mypath)

        print("{} is the file name",fn)
        print ("Accept incoming file:", fn)
        print ("Save it to:", mypath)
        #import tensorflow as tf
        import numpy as np
        from tensorflow.keras.preprocessing import image

        from tensorflow.keras.models import load_model
        # mypath=r"D:\RICE LEAF DISEASE DETECTION\imges\Blast_IMG_0_32.jpg"


        new_model = load_model(r"D:\RICE LEAF DISEASE DETECTION\BASE PAPER CODE\alg\MobileeNt.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)

        # prediction=np.argmax(result)
        prediction = classes[np.argmax(result)]
        if prediction==1:
            msg='Remedied of Rice LeafBlast :: fungicides like triazoles and strobilurins can be used judiciously for control to leaf blast.'
            # prediction=classes[np.argmax(result)]
        elif prediction==0:
            msg='Remedies of BrownSpot :: fungicides(e.g., iprodione, propiconazole, azoxystrobin, trifloxystrobin, and carbendazim) & Treat seeds with hot water.'
            # prediction=classes[np.argmax(result)]
        elif prediction==2:
            msg='Remedies of LeafBlight :: Seed treatment with bleaching powder & Seed treatment - seed soaking for 8 hours in Agrimycin (0.025%) and wettable ceresan (0.05%) followed by hot water treatment for 30 min at 52-54oC'
            # prediction=classes[np.argmax(result)]
        else:
            msg=''
    return render_template("template.html",image_name=fn, text=prediction ,msg=msg)
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("D:/RICE LEAF DISEASE DETECTION/imges", filename)
@app.route('/upload1')
def upload1():
    return render_template("upload.html")


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run()