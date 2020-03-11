Follow the running sequence: 

0. Using weight file/model file that is in previous facenet folder if using facenet

OR - Using dlib

Follow these steps after conda activate the ml1p13 env. pip install dlib

pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f

pip install face_recognition

1. Running facenet_predict5.py (facenet)
   OR - Running facenet_predict6.py (dlib)
   
2. Running KNN.py, mlp.py, logistic_regression.py (facenet)
   OR - Running KNN_dlib.py, mlp_dlib.py , logistic_regression_dlib.py , svm_dlib.py (dlib)
   
3. Running voting.py
   OR - Running voting_dlib.py 
   
4. Running webcam_cv3_facenet3b.py
   OR - Running webcam_cv3_dlib.py
