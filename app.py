from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import numpy as np 
import pickle
from keras.models import load_model

from keras import backend as K
import os


# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

with open('models/tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)




app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
	max_length = 200
	model = load_model('models/my_model.h5')
    # Loading our AI Model
	if request.method == 'POST':
		namequery = request.form['text']
		data = [namequery]

		tokenizer.fit_on_texts(data)
		enc = tokenizer.texts_to_sequences(data)
		enc=pad_sequences(enc, maxlen=max_length, padding='post')
		my_prediction = model.predict(enc)
		my_prediction=my_prediction[0]

		K.clear_session()
	return render_template('result.html',prediction = my_prediction[0])
	return render_template('result.html')
	# K.clear_session()



if __name__ == '__main__':
	app.run(debug=True)



