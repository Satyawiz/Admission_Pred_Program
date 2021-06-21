from logging import debug
from flask import Flask,render_template, request
import joblib
import numpy as np

Adm_App=Flask(__name__)

model=joblib.load("Admission_Pred.pkl")

@Adm_App.route('/')
def welcome():
    return render_template('Admission.html')

@Adm_App.route('/predict' , methods = ['POST'])
def predict():
    gre=request.form.get('gre')
    toefl=request.form.get('toefl')
    unvrate= request.form.get('unvrate')
    sop=request.form.get('sop')
    lor=request.form.get('lor')
    cgpa=request.form.get('cgpa')
    res=request.form.get('radio')
    prediction =model.predict([[int(toefl),int(unvrate),int(sop),int(lor),int(cgpa),int(res)]])
    output = np.round(prediction[0][0] , 2)*100
    return render_template('Admission.html' , prediction_text = f"The candidate will be having {output} % of admission chances")




if __name__=='__main__':
    Adm_App.run(debug=True)