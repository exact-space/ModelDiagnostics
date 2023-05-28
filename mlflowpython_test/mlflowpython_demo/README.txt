Run these commands:


1. Install pre requisites and postgres
a. sudo apt install gcc
b. sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all

2. Create python conda env with required libraries
a) conda create --name mlflow_demo python=3.9
b) conda activate mlflow_demo
c) conda install pip
d) conda install -p /home/user/miniconda3/envs/mlflow_demo ipykernel --update-deps --force-reinstall
e) pip install -r requirements.txt


3. Connect to postgres and do one time activities
a. sudo -u postgres psql
b. Run these in the postgres prompt
CREATE DATABASE mlflow;
CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

4. Do mlflow one time activities
a. mkdir ~/mlruns

5. Run mlflow. Open command prompt and run these
a. conda activate mlflow_demo
b. mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root file:/home/mlruns -h *.*.*.* -p 5000