from flask import Flask, jsonify, request, send_file
import os
from flask_cors import CORS

def classify(file, option):
    file_path = None
    if option == 'Random Forest':
        file_path = os.path.join('uploads', "RandomForest.csv")
        file = randomForestClassification(file)
    else:
        file_path = os.path.join('uploads', "DecisionTree.csv")
        file = decisionTreeClassification(file)

    return file_path

def randomForestClassification(file):
    return file

def decisionTreeClassification(file):
    return file

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    option = request.form['option']
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

    file_path = classify(file, option)

    return send_file(file_path, as_attachment=True)

app.run(port=5001, host='localhost', debug=True)

#############################################
