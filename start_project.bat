@echo off

start start_alert_service.bat

sleep 1

start start_security_access_monitor.bat

sleep 1

start start_intruder_detection.bat
