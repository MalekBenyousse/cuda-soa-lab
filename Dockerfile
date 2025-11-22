FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8140

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8140"]
