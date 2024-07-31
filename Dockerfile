FROM python:3.11-slim

EXPOSE 5000/tcp

WORKDIR /PhishEye/

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# CMD ["flask", "run", "--host", "0.0.0.0" ]
CMD ["bash", "-c", "python app.py --model_type ${MODEL_TYPE}"]