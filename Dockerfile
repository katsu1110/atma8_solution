FROM jupyter/datascience-notebook

COPY requirements.txt .

RUN pip install -U pip && \
    pip install -r requirements.txt -U && \