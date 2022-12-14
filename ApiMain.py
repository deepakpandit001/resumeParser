from ResumeParseMethods import *
from flask import render_template, request
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    args = request.args
    path = args.get("path")
    getPdfFromUrl(path)
    pdfText = pdfToJson("resumeParse.pdf")
    return pdfText


@app.route('/get')
def isWorking():

    return "Hay"


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
