
#### **Capstone project: Intruder Detection and Face Recognition**


**Prerequisites:**

1. python version >= 3.6

2. conda version >= 4.6

For the details of python libraries which are being used, please refer to the environment.yml.


**Key techniques being used:**

1. deep learning, MTCNN, Haar Cascade

2. KNN, MLP, Logistic Regression and SVM classification models, and voting ensemble of these

3. opencv and background subtraction with both mog2 and knn

4. python REST API based on flask, twilio real-time sms and email alert


**The whole project includes following 3 parts:**

1. intruder detection

2. face recognition

3. sms and email alert


**How to setup the project from source**

1. git clone or download this repo to your local

2. open this capstone4 folder with any python IDE, then choose the env created above.

3. the source codes are included in following folder accordingly:

3.1 backend_service, source codes for sms/email alert and also the communication codes for diff modules

3.2 face_recognition_api, source codes for face recognition   

3.3 intruder_detection, source codes for intruder detection


**How to start this project as a package:**

1. create a conda env with the ml1p13.yml mentioned above.

2. git clone or download this repo to your local

3. open cmd window and activate the the installed environment.

4. go to the project folder like below:

4.1 cd c:/user/gary/capstone4/

4.2 execute start_project.bat to start the project. This will run all the modules.


**How to start each individual module as a package:**

1. start_security_access_monitor.bat, used to start alert service and backend communication api.

2. start_intruder_detection.bat, used to start intruder detection service.

3. start_security_access_monitor.bat, used to start face recognition service.