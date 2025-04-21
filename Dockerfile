FROM python:3.13-bookworm as stego
WORKDIR /opt/stego
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
