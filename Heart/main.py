from fastapi import FastAPI
import pickle as pk
app=FastAPI()

@app.get('/inference/')
def inference(input):
    model=pk.load(open("Heart/model.pkl","rb"))
    response=model.predict(input)
    if response==0:
        return "Has No Heart"
    else:
        return "Has Heart"

    