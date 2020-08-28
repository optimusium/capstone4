
**Function of the code**

Detect movement in webcam video through the Background Subtraction method. Any detected movement will be display in white against a black background.


**How to run it directly via code**

Following input parameters need to be specified in run/debug configuration
--input 0/1, 0 means integrated camera, 1 means external connected camera
--algo MOG2/KNN, different ways of background subtraction

example: --input 0 algo KNN 


**How to run it via python console**

1. open cmd window and activate the the installed environment.

2. go to the project folder like below:

    2.1 cd c:/user/gary/capstone4/
    
    2.2 execute start_intruder_detection.bat.