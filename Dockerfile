FROM python:3.8.12-slim

COPY package_folder package_folder
COPY requirements.txt requirements.txt
COPY models models
COPY setup.py setup.py

RUN pip install -e .

#RUN CONTAINER LOCALLY!
# CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0

#RUN CONTAINER DEPLOYED
CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0 --port $PORT
