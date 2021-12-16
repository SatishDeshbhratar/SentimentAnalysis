import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import traceback

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
        comment = [np.array(str(request.form.values()))]
        print(jsonify(comment))
        prediction = model.predict(comment)

    except Exception:
        traceback.print_exc()

    return render_template('index.html', prediction_text='Sentiment analysis is - {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"] = True
