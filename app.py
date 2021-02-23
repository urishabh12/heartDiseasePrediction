from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS, cross_origin
import datetime
import json

#loading
dataset = pd.read_csv('dataset.csv')
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
#transforming data
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
#training set preparation
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
#training
knn_classifier = KNeighborsClassifier(n_neighbors = 8)
knn_classifier.fit(X_train, y_train)

#Global Dataset
dataset = {}

def readFromDatabase():
    global dataset
    with open('database.txt') as json_file:
        dataset = json.load(json_file)

def writeToDatabase():
    global dataset
    with open('database.txt', 'w') as outfile:
        json.dump(dataset, outfile)
    readFromDatabase()

readFromDatabase()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict/<string:username>', methods=['POST'])
@cross_origin()
def predict(username):
    data = request.json
    for key in data.keys():
        data[key] = [data[key]]
    df = pd.DataFrame.from_dict(data)
    ans = knn_classifier.predict(df)
    data['result'] = int(ans[0])
    data['dateTime'] = str(datetime.datetime.now())
    if username not in dataset['history'].keys():
        dataset['history'][username] = []
    dataset['history'][username].append(data)
    writeToDatabase()
    res = {'result': int(ans[0])}
    return jsonify(res)

@app.route('/user/add', methods=['POST'])
@cross_origin()
def addUser():
    data = request.json
    userDict = {}
    userDict['username'] = data['username']
    userDict['password'] = data['password']
    for i in dataset['users']:
        if(i['username'] == userDict['username']):
            return jsonify({'res': 'user exists'})
    dataset['users'].append(userDict)
    writeToDatabase()
    res = {'res': 'ok'}
    return jsonify(res)

@app.route('/user/login/<string:username>/<string:password>', methods=['GET'])
def login(username, password):
    ok = False
    for i in dataset['users']:
        if(i['username'] == username and i['password'] == password):
            ok = True

    res = {'valid': ok}
    return jsonify(res)

@app.route('/history/<string:username>')
def history(username):
    if(username not in dataset['history'].keys()):
        return jsonify({})
    return jsonify(dataset['history'][username])



app.run(threaded=True)
