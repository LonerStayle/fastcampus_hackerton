
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from app.core.config import EXAONE_3_0_7B_MODEL
from app.schema.MessageTransform import MessageTransformRequest, MessageTransformResponse
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from transformers import pipeline
from app.agents.transfer_agent import graph 

load_dotenv(override=True)
# --- 감정 분석용 싱글톤 클래스 ---
class EmotionAnalyzer:
    _instance = None
    _pipeline = None
    
    # KoElectra 모델의 라벨 매핑 (모델 학습 데이터 기준)
    # 0: 공포, 1: 놀람, 2: 분노, 3: 슬픔, 4: 중립, 5: 행복, 6: 혐오
    LABEL_MAP = {
        "0": "공포", "1": "놀람", "2": "분노", "3": "슬픔", 
        "4": "중립", "5": "행복", "6": "혐오",
        "LABEL_0": "공포", "LABEL_1": "놀람", "LABEL_2": "분노", "LABEL_3": "슬픔",
        "LABEL_4": "중립", "LABEL_5": "행복", "LABEL_6": "혐오"
    }
    

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            print("Loading Emotion Model (CPU)...")
            # device=-1은 CPU를 의미합니다.
            cls._pipeline = pipeline(
                "text-classification", 
                model="dlckdfuf141/korean-emotion-kluebert-v2", 
                device=-1, 
                top_k=1
            )
        return cls._pipeline

    @classmethod
    def analyze(cls, text: str):
        classifier = cls.get_pipeline()
        result = classifier(text)[0][0]  # top_k=1이므로 첫번째 결과만 가져옴
        
        raw_label = str(result['label'])
        score = result['score']
        
        # 라벨을 한글로 변환 (매핑에 없으면 원본 사용)
        korean_emotion = cls.LABEL_MAP.get(raw_label, raw_label)
        
        # 퍼센트 포맷팅 (예: 98.5%)
        confidence_str = f"{score * 100:.1f}%"
        
        return korean_emotion, confidence_str

# model = None
# tokenizer = None

# def load_model():
#     global model, tokenizer

#     if model is None:
#         model_name = EXAONE_3_0_7B_MODEL

#         if torch.cuda.is_available():
#             device_map = "cuda"
#         elif torch.backends.mps.is_available():
#             device_map = "mps"
#         else:
#             device_map = "cpu"
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",      
#             bnb_4bit_compute_dtype=torch.float16  
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             quantization_config=bnb_config,
#             device_map=device_map
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# async def transform_message(request: MessageTransformRequest):
#     load_model()

#     # Prepare messages
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": request.message}
#     ]
#     text_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
#     encodings = tokenizer(text_prompt, return_tensors="pt")
#     input_ids = encodings["input_ids"].to("cuda")
#     output = model.generate(
#         input_ids,               # 첫 번째 인자는 무조건 Tensor여야 합니다.
#         max_new_tokens=128,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=False
#     )

#     full_output = tokenizer.decode(output[0], skip_special_tokens=False)

#     if "[|assistant|]" in full_output:
#         transformed = full_output.split("[|assistant|]")[-1].strip()
#         for token in ["[|endofturn|]", "</s>"]:
#             transformed = transformed.replace(token, "").strip()
#     else:
#         transformed = tokenizer.decode(output[0], skip_special_tokens=True)
#         if request.message in transformed:
#             transformed = transformed.split(request.message)[-1].strip()

#     return MessageTransformResponse(
#         original_message=request.message,
#         transformed_message=transformed
#     )

async def transform_message(request: MessageTransformRequest):
    response = await graph.ainvoke({
        "messages": [request.message]
    })
    recommendation = response.get("recommendation", "")
    answer = response["messages"][-1].content

    # 2. 추가 로직: 감정 분석 실행
    # (원본 메시지의 감정을 파악하는 것이 일반적이나, 변환된 메시지를 분석하려면 request.message 대신 answer를 넣으세요)
    emotion, confidence = EmotionAnalyzer.analyze(request.message)

    return MessageTransformResponse(
        original_message=request.message,
        transformed_message=answer,
        emotion=emotion,
        confidence=confidence,
        recommendation=recommendation
    )

    
    

    