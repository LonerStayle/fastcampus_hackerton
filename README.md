# 실행
'uv run uvicorn app.main:app --reload'

# (서버환경)배포 
'nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &'