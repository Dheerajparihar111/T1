from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# THIS CREATES PARAMETERS IN SWAGGER
class UserRequest(BaseModel):
    user_id: str


@app.get("/")
def home():
    return {"message": "Working"}


@app.post("/predict")
def predict(request: UserRequest):

    return {
        "received_user_id": request.user_id
    }