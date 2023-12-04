# Implement an NCI cancer incidence lookup API as a web server in Flask
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# Main/index page with form
@app.route("/")
def index():
    return render_template("index.html")


# Info page with results
@app.route("/info", methods=["POST"])
def info():
    state = request.form['state'].lower()
    analysis = state.upper()
    return render_template("info.html", state=state, analysis=analysis)


# API endpoint for state
@app.route("/state/<string:name>")
def state(name):
    data = {"names": ["John", "Jacob", "Julie", "Jennifer"]}
    return jsonify(data)


# Prevents execution if imported
if __name__ == '__main__':
    app.run(debug=True, ssl_context=("cert.pem", "key.pem"), port=5000)
