# Implement web server in Flask
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info", methods=["POST"])
def info():
    pass


@app.route("/state/<string:name>")
def state(name):
    data = {"names": ["John", "Jacob", "Julie", "Jennifer"]}
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, ssl_context=("cert.pem", "key.pem"), port=5000)
