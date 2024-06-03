from fastapi import FastAPI
import pickle

app = FastAPI()

@app.get('/')
def root():
    return {'greeting': 'bonsoir'}

@app.get('/predict')
def predict(sepal_length=1,
            sepal_width=1,
            petal_length=1,
            petal_width=1):
    with open('models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction_ = float(prediction[0])
    return {"prediction": prediction_}
