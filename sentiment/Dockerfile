FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt update

RUN apt -y upgrade

RUN apt -y install git-core python3 pip

RUN git clone https://github.com/huggingface/transformers.git

WORKDIR /transformers

RUN git checkout d1f74b9aff71a1d5a949b0bc48dbe9894cc6e40b

COPY patch.diff /transformers

RUN git apply patch.diff

RUN pip install -e .

RUN pip install pandas torch scikit-learn uvicorn fastapi 

COPY sentiment_rest.py /transformers/examples/text-classification

WORKDIR /transformers/examples/text-classification

CMD uvicorn sentiment_rest:app --host 0.0.0.0
