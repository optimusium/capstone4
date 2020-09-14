@echo off

set PYTHONPATH=%PYTHONPATH%;%cd%

cd backend_service

python app.py
