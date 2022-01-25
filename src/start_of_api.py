import torch
import rasterio
import xarray
import einops

import numpy as np
import pandas as pd


import logging
from fastapi import FastAPI
from models.ViT import *

app = FastAPI(title="Wildfire Cost Explorer", description="API for exploring causal factors in Wildfire", version="1.0")

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='logs.log')

model = None

@app.on_event("startup")
def load_model():
    global model
    model_fire = torch.load("model.pth").to('CPU')
    model_hp = torch.load("model_hp.pth").to('CPU')

@app.post("/api", tags=["prediction"])
async def get_predictions(vit: ViT):
    try:
        data.to('CPU'), tabular.to('CPU') = dat_in

        prediction = list(model_fire.predict(data, tabular))
        prediction_hp = list(model_hp.predict(data, tabular))
        
        return {"normed fire cost prediction": prediction, 'predictable property value': prediction_hp}
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}