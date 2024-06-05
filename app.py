from flask import Flask, request, make_response, jsonify
from flask_pymongo import PyMongo

from config import prod_db
from DataHandler import DataHandler

from endpoints import endpoints

import os
os.environ['FLASK_ENV'] = 'development'

def create_app(config=None):
    app = Flask(__name__)
    app.config["MONGO_URI"] = "mongodb://localhost:27017/{}".format(prod_db)
    mongo = PyMongo(app)
    db = mongo.db

    handler = DataHandler(db)

    endpoints(app, handler)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)