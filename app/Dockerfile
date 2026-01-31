# 1. 파이썬 버전 지정 (가벼운 slim 버전 추천)
FROM python:3.12-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 라이브러리 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. 서버 실행 (FastAPI 기본 포트 8000)
# Docker 안에서는 0.0.0.0으로 열어야 외부에서 접속 가능
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]