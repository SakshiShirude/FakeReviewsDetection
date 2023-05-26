from flask import Flask, jsonify, render_template, request,redirect,url_for
import numpy as np
import pandas as pd
import sklearn as sk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1692057600<0 else 0
from train_model import trainModel
import os
from werkzeug import secure_filename



app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def login():
	error = None
	if request.method == 'POST':
		if request.form['username'] != 'admin' or request.form['password'] != 'admin':
			error = 'Invalid Credentials. Please try again.'
		else:
			return redirect(url_for('testing'))

	return render_template('login.html', error=error)

@app.route('/home')
def root():
	return redirect(url_for('login'))


@app.route('/testing', methods=['GET','POST'])
def testing():
	global x
	global X
	df = pd.read_csv('deceptive-opinion.csv')
	df1 = df[['deceptive', 'text']]
	df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
	df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
	X = df1['text']
	Y = np.asarray(df1['deceptive'], dtype = int)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109)
	cv = CountVectorizer()
	x = cv.fit_transform(X_train)
	y = cv.transform(X_test)
	if request.method == 'POST':
		message = request.form['packet']
		data = [message]
		vect = cv.transform(data).toarray()
		pred = model.predict(vect)

		if(pred[0] == 1):
			pred = "This is Not a Spam Review..!!!"
		else:
			pred = 'This is the Spam Review...!!!'

		return render_template('testing.html', prediction_text=pred,packet=message)
	return render_template('testing.html')

@app.route('/train')
def train():
	acc1,acc2 = trainModel()
	return render_template('train.html',acc1=acc1*100,acc2=acc2*100)


@app.route('/result', methods=['GET', 'POST'])
def result():
	global x
	global X
	if request.method == 'POST':
		if request.form['sub']=='Show Result':
			result = model.predict(x)
			print(result)
			spam = np.count_nonzero(result==0)
			print(spam)
			nonspam = len(X)-spam
			result = np.where(result == 1, 'Not Spam', result)
			result = np.where(result == '0', 'Spam', result)
			print(result)
			df1 = pd.DataFrame(X)
			df2 = pd.DataFrame(result)
			df = pd.concat([df1, df2], axis=1, join='inner')
			print(df)
			df.to_csv('Result.csv')
			return render_template('result.html', spam=spam,nonspam=nonspam,tables=[df.head(10).to_html(classes='w3-table-all w3-hoverable')], titles=df.columns.values)
	return render_template('result.html')

if __name__ == '__main__' and run:
	app.run(debug=True)