@echo off
cd backend_service

start start_alert_service.bat


cd ..


start Start_security_access_monitor.bat


start Start_intruder_detection.bat

