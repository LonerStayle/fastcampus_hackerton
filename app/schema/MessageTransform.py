from pydantic import BaseModel
class MessageTransformRequest(BaseModel):
    message: str

class MessageTransformResponse(BaseModel):
    original_message: str
    transformed_message: str