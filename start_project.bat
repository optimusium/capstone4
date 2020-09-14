@echo off

start start_alert_service.bat

TIMEOUT /T 1

start start_security_access_monitor.bat

TIMEOUT /T 1

start start_intruder_detection.bat
