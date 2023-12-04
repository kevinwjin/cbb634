# Implement an NCI cancer incidence lookup API as a web server in Flask
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
    data = pd.read_csv("incidence_rates.csv")
    # rate = float(data[data["State"].str.contains(name.capitalize())].iloc[:, 2])
    return jsonify(data[name])


# Info page with results
@app.route("/info/", methods=["GET"])
def info():
    state = request.args.get("state")
    analysis = state.capitalize()
    return render_template("info.html", state=state, analysis=analysis)


# Prevents execution if imported
if __name__ == '__main__':
    app.run(debug=True,  # Allows for verbose error messages
            ssl_context=("cert.pem", "key.pem"),  # SSL cert/key
            host='0.0.0.0',  # Allows external connections
            port=5000)  # Port to run on
