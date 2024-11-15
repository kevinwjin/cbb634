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
    if name == "US" or name == "U.S." or name == "United States":
        name = "US"
    elif name == "DC" or name == "D.C." or name == "Washington DC" or name == "Washington D.C.":
        name = "District of Columbia"
    elif name not in data["State"]:
        return render_template("error.html")
    else:
        name.capitalize()
    entry = dict(data[data["State"].str.contains(name)].iloc[0, [1, 2]])
    return jsonify(entry)


# Info page with results
@app.route("/info/", methods=["GET"])
def info():
    state = request.args.get("state")
    data = pd.read_csv("incidence_rates.csv")
    if state == "U.S." or state == "United States":
        state = "US"
    elif state == "DC" or state == "D.C." or state == "Washington DC" or state == "Washington D.C.":
        state = "District of Columbia"
    elif state not in data["State"]:
        return render_template("error.html")
    else:
        state.capitalize()
    rate = float(data[data["State"].str.contains(state)].iloc[0, 2])
    return render_template("info.html", state=state, rate=rate)


# Prevents execution if imported
if __name__ == '__main__':
    app.run(debug=True,  # Allows for verbose error messages
            ssl_context=("cert.pem", "key.pem"),  # SSL cert/key
            host='0.0.0.0',  # Allows external connections
            port=5000)  # Port to run on
