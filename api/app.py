# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from configs import cols
import sys
sys.path.insert(-1, '../code')
from model import titanicModel



clf_path = '../models/TitanicClassifier.pkl'
with open(clf_path, 'rb') as f:
    Cmodel = titanicModel(pickle.load(f))

# initialize a flask application
app = Flask("will-they-survive")

#
@app.route("/", methods=["GET"])
def hello():
    return jsonify("hello from ML API of Titanic data!")

@app.route("/predictions", methods=["GET"])
def predictions():
    clf_path = '../models/TitanicClassifier.pkl'
    with open(clf_path, 'rb') as f:
        Cmodel = titanicModel(pickle.load(f))
    # df = pd.read_json('data.json')
    data = request.get_json()
    df=pd.DataFrame(data['data'])
    data_all_x_cols = cols
    loaded_model = pickle.load(open('../models/TitanicClassifier.pkl', 'rb'))
    try:
        preprocessed_df=Cmodel.prepare(df)
    except:
        return jsonify("Error occured while preprocessing your data for our model!")
    try:
        predictions= Cmodel.predict(preprocessed_df[data_all_x_cols])
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    response={'data':[],'prediction_label':{'survived':int(1),'not survived':int(0)}}
    response['data']=list(predictions)
    return make_response(jsonify(response),200)

if __name__ == '__main__':
        app.run(debug=True)