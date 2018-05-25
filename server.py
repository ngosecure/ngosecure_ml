#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import os

app = Flask(__name__)
api = Api(app)

class InvestmentType(Resource):
    def get(self):
        os.system('python NGO-RISK-PREDICTION.py')
        data = pd.read_csv("NGO-RISK-PREDICTION.csv")
        return jsonify(data.to_json())

api.add_resource(InvestmentType, '/ngo/') # Route_4

if __name__ == '__main__':
     app.run()