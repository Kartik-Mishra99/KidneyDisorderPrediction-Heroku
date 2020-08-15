import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

model = pickle.load(open('RF_kidney.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	age = int(request.form['age'])
    	bp = float(request.form['bp'])
    	al = int(request.form['al'])
    	pcc = int(request.form['pcc'])
    	bgr = float(request.form['bgr'])
    	bu = float(request.form['bu'])
    	sc = float(request.form['sc'])
    	hemo = float(request.form['hemo'])
    	pcv = int(request.form['pcv'])
    	htn = int(request.form['htn'])
    	dm = int(request.form['dm'])
    	appet = int(request.form['appet'])

    	data = np.array([[age,bp,al,pcc,bgr,bu,sc,hemo,pcv,htn,dm,appet]])

    	preds = model.predict(data)
    	return render_template('result.html',prediction=preds)
    return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)



