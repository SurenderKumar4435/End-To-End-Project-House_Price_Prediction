from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.tree import DecisionTreeRegressor

application  = Flask(__name__)
app = application

## import pkl file----------------->>>>>>>>>
regress_model = pickle.load(open("model.pkl","rb"))
standard_scaler = pickle.load(open("scaler.pkl","rb"))


## Route for home page ---------->>>>>>>>>>>
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
         Pregnancies = float(request.form.get("Pregnancies"))
         Glucose= float(request.form.get("Glucose"))
         BloodPressure = float(request.form.get("BloodPressure"))
         SkinThickness = float(request.form.get("SkinThickness"))
         Insulin = float(request.form.get("Insulin"))
         BMI = float(request.form.get("BMI"))
         DiabetesPedigreeFunction= float(request.form.get("DiabetesPedigreeFunction"))
         Age = float(request.form.get("Age"))
         
         
         

         new_data_scaled = standard_scaler.transform([[PassengerId,Pclass,Age,SibSp,Parch,Fare,Sex_female,Sex_male,Embarked_C,Embarked_Q,Embarked_S]])
         result = regress_model.predict(new_data_scaled)

         return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True)

