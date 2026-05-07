from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router
from fastapi import APIRouter
from implementation import Testing
from models.FlowerInput import FlowerInput


approuter = APIRouter(prefix="/predict", tags=["predict"])
testing = Testing()

origins = [
    "http://localhost:4200"
]


app = FastAPI(
    title="Practica4 - Web"
)

@approuter.get("/")
def getMain():
    return {
        "Message": "Api Funcionando"
    }


@approuter.post("/")
def predictionAPI(data: FlowerInput):
    result = testing.predict_flower(
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    )
    return result


#test = Testing()
#my_flower = [1.2, 3.4, 2.1, 5.1]
#prediccion = test.predict_flower(1.2, 3.4, 2.1, 5.2)
#print(prediccion)



app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)





#app.include_router(router)
app.include_router(approuter)
