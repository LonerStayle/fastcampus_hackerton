# from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import ai_router, test_router
from fastapi.middleware.cors import CORSMiddleware
# from app.core.database import init_vector_extension


# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     await init_vector_extension()
#     yield


app = FastAPI()
app.include_router(test_router.router)
app.include_router(ai_router.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 이 부분이 5173 포트 허용
    allow_credentials=True,  # 쿠키/인증 헤더 허용 (필요 시)
    allow_methods=["*"],     # 모든 HTTP 메서드 (GET, POST 등) 허용
    allow_headers=["*"],     # 모든 헤더 허용
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the fastcampus-hackathon API!"}

