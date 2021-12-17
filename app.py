import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import traceback
import pandas as pd
# Libraries for feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        comment = [str(request.form['inputText'])]
        print(comment)
        reshaped_comment = [comment]
        print(reshaped_comment)

        # build BOW features on train reviews
        cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1,2))
        cv_train_features = cv.fit_transform(reshaped_comment)

        # build TFIDF features on train reviews
        tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
        tv_train_features = tv.fit_transform(reshaped_comment)

        prediction = model.predict(reshaped_comment)

    except Exception:
        traceback.print_exc()

    return render_template('index.html', prediction_text='Sentiment analysis is - {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"] = True
