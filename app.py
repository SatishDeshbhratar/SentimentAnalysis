from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tv_transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		confidence_score = clf.predict_proba(vect)
		if my_prediction == 1:
			confidence_score = round(confidence_score[0][1] * 100,2)
		else:
			confidence_score = round(confidence_score[0][0] * 100,2)
	return render_template('result.html',prediction = my_prediction, confidence = confidence_score)

if __name__ == '__main__':
	app.run(debug=True)