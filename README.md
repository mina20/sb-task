# sb-task
assignment from smart-bricks 
This project serves a trained meta-learner model using FastAPI. The model is trained with PyTorch and exposed via an API to make predictions.
The data used for this model is prediction from base models. Here we used randomForest, XGboost and svr.
The data preprocessing and feature engineering is performed before getting the prediction from base model.
To run the model and get prediction please follow the steps
1. Clone the repo using
git clone https://github.com/mina20/sb-task.git
2. navigate to sb-task
3. built docker using docker build -t fastapi-meta-learner .
4. run docker using docker run -d -p 8000:8000 fastapi-meta-learner
5. use test.py to send a post request.
# Direct use without docker
1. run the command uvicorn main:app
2. use test.py to post request.