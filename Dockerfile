FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN sudo apt-get update 
RUN sudo apt-get install build-essential -y
RUN sudo apt-get install cmake -y
RUN sudo apt-get install libx11-dev -y
RUN sudo apt-get install gcc -y

RUN sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda

RUN pip install -r requirements.txt

COPY . /opt/app

CMD ["python", "main.py"]
