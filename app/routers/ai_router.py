
from fastapi import APIRouter, HTTPException
from app.schema.MessageTransform import MessageTransformRequest, MessageTransformResponse
from app.service import transfer_service
import random

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post("/transform", response_model=MessageTransformResponse)
async def transform_message(request: MessageTransformRequest):
    # try:
    result = await transfer_service.transform_message(request)
    return result

swear_list = [
    "야 이 새끼야, 왜 전화 안 받아? 빨리 환불 처리해!"
    "미친년아, 주소 잘못 입력됐잖아. 당장 고쳐!"
    "개자식아, 이 제품 개쓰레기야. 돈 돌려줘!"
    "입 다물고 그냥 해, 병신아. 이름 물어보지 마!"
    "야 좆같은 놈아, 왜 이렇게 느려? 빨리 해!"
    "씨발, 이 회사 미쳤어? 환불 왜 안 돼?"
    "야 멍청이 새끼야, 주소 바꾸라고 했잖아!"
    "닥쳐, 개년아. 그냥 꺼지라고!"
    "미친놈아, 이름 뭐냐고? 환불이나 빨리 해!"
    "야 이 병신아, 왜 대기시키냐? 빨리 나와!"
    "새끼야, 이 제품 왜 안 와? 사기 치냐?"
    "입 닥치고 처리해, 미친년아. 주소 고쳐!"
    "야 개자식아, 돈 돌려줘. 안 그러면 고소할 거야!"
    "병신같은 놈아, 왜 전화 끊어? 다시 해!"
    "씨발, 이 서비스 개판이네. 환불 해!"
    "야 멍청아, 이름 확인 왜 해? 빨리 주소 바꿔!"
    "닥치고 꺼져, 새끼야. 더 이상 말 마!"
    "미친놈아, 제품 불량이잖아. 돈 줘!"
    "야 이 년아, 왜 이렇게 느려? 환불 처리해!"
    "개자식아, 주소 잘못됐어. 당장 수정해!"
    "입 다물고 해, 병신아. 이름 물어보지 마!"
    "야 좆같은 새끼야, 왜 안 돼? 빨리!"
    "미친년아, 이 회사 사기야? 환불 해!"
    "씨발, 대기 시간 왜 이렇게 길어? 나와!"
    "야 멍청이야, 제품 왜 안 보내? 돈 돌려!"
    "닥쳐, 개년아. 그냥 처리하라고!"
    "미친놈아, 주소 바꾸라고 했잖아!"
    "야 병신아, 이름 뭐냐? 환불이나 해!"
    "새끼야, 이 서비스 개쓰레기네. 꺼져!"
    "입 닥치고 빨리 해, 미친놈아. 주소 고쳐!"
]

@router.get("/swear")
async def get_swear():
    selected_swear = random.choice(swear_list)
    return {"swear": selected_swear}    
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

