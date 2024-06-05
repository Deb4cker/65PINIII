from flask import Flask, jsonify, request, send_file
import os
import classificationService as service
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify(file, option):
    # Leia o arquivo CSV e transforme-o em um DataFrame
    data = pd.read_csv(file)

    # Remova a coluna 'id' ou outras colunas não necessárias
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    if option == 'Random Forest':
        predictions = randomForestClassification(data)
        # Salve as previsões em um novo CSV
        data['classe'] = predictions
        file_path = 'uploads/random_forest.csv'
        data.to_csv(file_path, index=False)
    else:
        predictions = decisionTreeClassification(data)
        # Salve as previsões em um novo CSV
        data['classe'] = predictions
        file_path = 'uploads/decision_tree.csv'
        data.to_csv(file_path, index=False)

    return file_path


def randomForestClassification(data):
    return service.randomForestClassification(data)


def decisionTreeClassification(data):
    return service.decisionTreeClassification(data)


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

    # Garanta que o arquivo foi salvo corretamente antes de classificá-lo
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Reabra o arquivo salvo para processamento
    with open(file_path, 'r') as f:
        result_path = classify(f, option)

    return send_file(result_path, as_attachment=True)


if __name__ == '__main__':
    app.run(port=5001, host='localhost', debug=True)
