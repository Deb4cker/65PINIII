from flask import Flask, jsonify, request, send_file
import os
import random

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#getAll
@app.route('/getFile', methods=['GET'])
def get_file():
    # Neste exemplo, retornamos um arquivo de exemplo chamado 'exemplo.csv' na pasta 'uploads'
    file_path = os.path.join('uploads', 'TesteAPi.csv')

    # Verifica se o arquivo existe
    if os.path.exists(file_path):
        # Retorna o arquivo como parte da resposta
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'Arquivo não encontrado'}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifica se o arquivo foi enviado na requisição
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    # Verifica se o nome do arquivo é vazio
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio'}), 400

    # Verifica se o arquivo possui uma extensão permitida
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

    # Salva o arquivo no servidor (opcional)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Retorna o arquivo como parte da resposta
    return send_file(file_path, as_attachment=True)


app.run(port=5000, host='localhost', debug=True)
