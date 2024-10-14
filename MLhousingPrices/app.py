from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
from MLhousingPrices import preprocessor, model
import io
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    return {"filename": file.filename}

@app.post("/predict") 
def prediction(file: UploadFile = File(...)):
    X_pred = pd.read_csv(file.file)
    file.file.close()

    pp = preprocessor.load_preprocessor()
    X_pred_preprocessed = pp.transform(X_pred)

    lr = model.load_model()
    y_pred = lr.predict(X_pred_preprocessed)
    X_pred_preprocessed['predictions'] = y_pred

    stream = io.StringIO()
    X_pred_preprocessed.to_csv(stream, index = False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)