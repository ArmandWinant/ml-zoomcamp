FROM python:3.9.6-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_decisiontree.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind",  "0.0.0.0:9696",  "predict:app" ]