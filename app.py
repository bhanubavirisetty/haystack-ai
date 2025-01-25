from fastapi import FastAPI,Request,Response,Form
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import json
from dotenv import load_dotenv

app=FastAPI()
templates=Jinja2Templates(directory="templates")


@app.get("/")
async def index(request:Request):
    return templates.TemplateResponse("index.html",{"request":Request})


@app.post("/get_answer")
async def get_answer(request:Request,questions:str=Form(...)):
    pass

