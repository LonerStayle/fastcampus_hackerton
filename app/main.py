from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import test, exaone
from app.core.database import init_vector_extension


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_vector_extension()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(test.router)
app.include_router(exaone.router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the fastcampus-hackathon API!"}

