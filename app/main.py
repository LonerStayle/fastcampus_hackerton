from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import ai_router, test_router
# from app.core.database import init_vector_extension


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     await init_vector_extension()
#     yield


app = FastAPI()
app.include_router(test_router.router)
app.include_router(ai_router.router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the fastcampus-hackathon API!"}

