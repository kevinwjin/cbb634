# Server backend for final project
# Author: Kevin Jin
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)


# Main/index page with form
@app.route("/")
def index():
    return render_template("index.html")


# API endpoint for state
@app.route("/state/<string:name>")
def state(name):
    pass


# Info page with results
@app.route("/info/", methods=["POST"])
def info():
    pass


# Prevents execution if imported
if __name__ == '__main__':
    app.run(debug=True,  # Allows for verbose error messages
            ssl_context=("cert.pem", "key.pem"),  # SSL cert/key
            host='0.0.0.0',  # Allows external connections
            port=5000)  # Port to run on
