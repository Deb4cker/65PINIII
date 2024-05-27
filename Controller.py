from flask import Flask, jsonify, request, send_file
import os
import classificationService as service
from flask_cors import CORS

def classify(file, option):
    file_path = 'uploads/'
    if option == 'Random Forest':
        file_path += 'random_forest.csv'
        file = randomForestClassification(file)
        file.save(file_path)
    else:
        file_path += 'decision_tree.csv'
        file = decisionTreeClassification(file)
        file.save(file_path)

    return file_path

def randomForestClassification(file):
    return service.randomForestClassification(file)

def decisionTreeClassification(file):
    return service.decisionTreeClassification(file)

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
