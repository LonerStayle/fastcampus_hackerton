import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

router = APIRouter(prefix="/exaone", tags=["EXAONE"])

SYSTEM_PROMPT = """당신은 '상담원 보호 및 원활한 소통을 위한 AI 중재자'입니다.
사용자가 상담원에게 전달하려는 거칠거나 감정적인 메시지를 분석하여,
상담원이 상처받지 않으면서도 사용자의 요구사항을 정확히 파악할 수 있도록 '정중하고 전문적인 비즈니스 언어'로 변환하는 역할을 수행합니다.

### 지침:
1. **감정 제거**: 비속어, 모욕적인 표현, 비꼬는 말투, 지나친 느낌표나 분노 표현을 모두 삭제하세요.
2. **요점 추출**: 사용자가 불만을 느끼는 지점과 해결을 원하는 요구사항(핵심 본질)을 명확히 추출하세요.
3. **톤앤매너**: 공손하고 정중한 격식체(~합니다, ~해 주시기 바랍니다)를 사용하세요.
4. **객관화**: 사용자의 주관적인 분노를 "사용자가 ~한 부분에 대해 불편을 겪고 있음"과 같은 객관적인 서술로 바꾸세요.

### 변환 예시:
- 입력: "아니 도대체 배송 언제 오냐고! 사람 장난해? 당장 환불해줘!!"
- 변환: "배송이 언제 올까요? 즉각적인 환불 조치를 요청하겠습니다."
"""

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load EXAONE model and tokenizer"""
    global model, tokenizer

    if model is None:
        model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda"
        elif torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device_map in ["cuda", "mps"] else torch.float32,
            trust_remote_code=True,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)


class MessageTransformRequest(BaseModel):
    message: str
    max_new_tokens: Optional[int] = 128
    system_prompt: Optional[str] = None


class MessageTransformResponse(BaseModel):
    original_message: str
    transformed_message: str


@router.post("/transform", response_model=MessageTransformResponse)
async def transform_message(request: MessageTransformRequest):
    """
    Transform aggressive or emotional messages into polite, professional language.
    """
    try:
        load_model()

        system_prompt = request.system_prompt or SYSTEM_PROMPT

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]

        # Tokenize
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Get device
        device = next(model.parameters()).device

        # Generate
        output = model.generate(
            input_ids.to(device),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
        )

        # Decode
        full_output = tokenizer.decode(output[0], skip_special_tokens=False)

        # Extract only the assistant's response
        # The output contains the full conversation, we need to extract the generated part
        if "[|assistant|]" in full_output:
            transformed = full_output.split("[|assistant|]")[-1].strip()
            # Remove any trailing special tokens
            for token in ["[|endofturn|]", "</s>"]:
                transformed = transformed.replace(token, "").strip()
        else:
            # Fallback: use the full decoded output
            transformed = tokenizer.decode(output[0], skip_special_tokens=True)
            # Try to extract just the response part
            if request.message in transformed:
                transformed = transformed.split(request.message)[-1].strip()

        return MessageTransformResponse(
            original_message=request.message,
            transformed_message=transformed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transforming message: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if the model is loaded and ready"""
    return {
        "status": "ok" if model is not None else "not_loaded",
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model is not None else None
    }


@router.post("/load")
async def load_model_endpoint():
    """Manually load the model"""
    try:
        load_model()
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "device": str(next(model.parameters()).device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
