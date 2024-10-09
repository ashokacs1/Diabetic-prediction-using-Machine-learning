import pickle
from flask import Flask,request,render_template,app,jsonify
from flask import Response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import and load model and scaler pickle files 
model = pickle.load(open('Notebooks/Model.pkl','rb'))
scalar = pickle.load(open('Notebooks/Scalar.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predictdata',methods = ['GET','POST'])
def Diabetic_prediction():
    result=""
    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scalar.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('output.html',result=result)

    else:
        return render_template('home.html')

if __name__  == '__main__':
    app.run(host='0.0.0.0')