# 실행
'uv run uvicorn app.main:app --reload'

# (서버환경)배포 
'nohup sudo ./venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 80 > server.log 2>&1 &'