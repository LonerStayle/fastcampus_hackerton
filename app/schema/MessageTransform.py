from pydantic import BaseModel
class MessageTransformRequest(BaseModel):
    message: str

class MessageTransformResponse(BaseModel):

    original_message: str
    transformed_message: str
    emotion: str       # 추가됨: 감정 (예: "행복")
    confidence: str    # 추가됨: 확신도 (예: "99.2%")