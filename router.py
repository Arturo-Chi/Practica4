from fastapi import APIRouter
router = APIRouter(prefix="/model", tags=["model"])



@router.get("/")
def helloWorld():
    return {
        "message": "Hello World"
    }


@router.get("/asincronous")
def nothing():
    return {
        "Hello": "Hello",
        "data": [1, 2, 3, 4, 5, 6, 7]
    }

