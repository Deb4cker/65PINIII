from flask import Flask, jsonify, request

app = Flask(__name__)

users = [
    {
        'id': 1,
        'nome': 'Ana',
        'telephone': '555555'
    },
    {
        'id': 2,
        'nome': 'Nicolas',
        'telephone': '555555'
    },
]


#getAll
@app.route('/users', methods=['GET'])
def get_all_users():
    return jsonify(users)


#getById
@app.route('/users/<int:id>', methods=['GET'])
def get_by_id(id):
    for user in users:
        if user.get('id') == id:
            return jsonify(user)


#Edit
@app.route('/users/edit/<int:id>', methods=['PUT'])
def edit_user(id):
    new_user = request.get_json()
    for value, user in enumerate(users):
        if user.get('id') == id:
            users[value].update(new_user)
            return jsonify(users[value])


#Create
@app.route('/users/create', methods=['POST'])
def create_user():
    new_user = request.get_json()
    users.append(new_user)

    return jsonify(users)


#Delete
@app.route('/users/delete/<int:id>', methods=['DELETE'])
def delete_user(id):
    for value, user in enumerate(users):
        if user.get('id') == id:
            del users[value]

    return jsonify(users)


app.run(port=5000, host='localhost', debug=True)
