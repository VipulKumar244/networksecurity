import sys
import os

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import TARGET_COLUMN


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        from networksecurity.pipeline.training_pipeline import TrainingPipeline

        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)


@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        #print(df)
        final_model=load_object("final_model/model.pkl")
        if not isinstance(final_model, NetworkModel):
            raise TypeError("Saved model is not a NetworkModel")
        y_pred = final_model.predict(df)
        df['predicted_column'] = y_pred
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={"table": table_html},
        )
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)




if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=int(os.getenv("PORT", "7860")))
