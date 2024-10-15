from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
from MLhousingPrices import preprocessor, model
import io
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the MLhousingPrices FastAPI app, head to /docs to predict data and train models !"}

@app.post("/predict") 
def prediction(file: UploadFile = File(...)):
    """
     This endpoint uses the uploaded file to predict house_median_value using the tuned XGBRegressor model.  
     Upload the file via the button.  
     The response will contain a clickable 'Download file' button, which will include your input data and the predictions appended to it.
    """
    X_pred = pd.read_csv(file.file)
    file.file.close()

    pp = preprocessor.load_preprocessor()
    pp.preprocess_test(X_pred)

    lr = model.load_model()
    y_pred = lr.predict(pp.X_test_pp)
    predictions = pp.X_test_pp
    predictions['predictions'] = y_pred

    stream = io.StringIO()
    predictions.to_csv(stream, index = False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response

@app.post("/train") 
def train(file: UploadFile = File(...),  modelType = 'knn'):
    """
     This endpoint uses the uploaded file to train a model.  
     Set the modelType to either 'knn' (KNN Regressor), 'xgb' (XGBRegressor) or 'rfr' (RandomForestRegressor).  
     The model will be hypertuned, and as a response you will get a message containing the R2 score and the selected parameters.
    """
    data = pd.read_csv(file.file)
    file.file.close()

    pp = preprocessor.Preprocessor(data, cluster=False, subset=True) 
    pp.split().preprocess_train()

    mt = model.ModelTrainer(modelType=modelType)
    mt.train(X_train_pp=pp.X_train_pp, y_train=pp.y_train)

    return {"Best R2 score" : mt.model.best_score_, "Best params" : mt.model.best_params_}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)