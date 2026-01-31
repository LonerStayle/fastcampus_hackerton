
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from app.core.config import EXAONE_3_0_7B_MODEL
from app.schema.MessageTransform import MessageTransformRequest, MessageTransformResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from app.agents.transfer_agent import graph 
import os
load_dotenv(override=True)


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
    response = graph.invoke({
        "messages": [request.message]
    })
    answer = response["messages"][-1].content
    return MessageTransformResponse(
        original_message=request.message,
        transformed_message=answer
    )

    
    

    