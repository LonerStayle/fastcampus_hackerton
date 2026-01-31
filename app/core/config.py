
import torch

# 모델 설정
EXAONE_2_4_B_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# 하드웨어 설정 (맥북 MPS 대응)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"