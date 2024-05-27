import joblib

def randomForestClassification(data):
    
    # Carregar o modelo RandomForest
    model = joblib.load('models/random_forest_model.pkl')
    
    #criar arquivo csv com as previsões
    predictions = model.predict(data)

    return predictions

def decisionTreeClassification(data):
        
        # Carregar o modelo DecisionTree
        model = joblib.load('models/decision_tree_model.pkl')
        
        #criar arquivo csv com as previsões
        predictions = model.predict(data)
    
        return predictions