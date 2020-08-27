
**Capstone project: Intruder Detection and Face Recognition** 

Prerequisites:
1. python version >= 3.6
2. conda version >= 4.6
For the details of python libraries which are being used, please refer to the ml1p13.yml in env folder

The whole project includes following 3 parts:

1. intruder detection

2. face recognition

3. sms and email alert

~~For the details of each part, please go to according folders and check the readme there.~~

**How to setup the project from source**

1. git clone or download this repo to your local

2. open this capstone4 folder with any python IDE, then choose the env create above.

3. the source codes are included in following folder accordingly:

3.1 backend_service, source codes for sms/email alert and also the communication codes for diff modules

3.2 face_recognition_api, source codes for face recognition   

3.3 IntruderDetection, source codes for intruder detection


~~**How to run the project after setup**~~


**How to start this project as a package:**

1. create a conda env with the ml1p13.yml mentioned above.

2. git clone or download this repo to your local

3. open cmd window and go to the project folder like below:

3.1 cd c:/user/gary/capstone4/

3.2 execute Start_project.bat to start the project. This will run all the modules.

**How to start each individual module as a package:**

1. Start_security_access_monitor.bat, used to start alert service and backend communication api.

2. Start_intruder_detection.bat, used to start intruder detection service.

3. Start_face_recognition.bat, used to start face recognition service.