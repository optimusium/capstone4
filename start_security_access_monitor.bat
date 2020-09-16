@echo off

set PYTHONPATH=%PYTHONPATH%;%cd%

cd face_recognition_api

python security_access_monitor.py --input 0