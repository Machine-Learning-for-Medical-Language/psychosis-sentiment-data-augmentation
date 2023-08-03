FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ENV HTTP_PROXY="http://10.41.13.6:3128"
ENV HTTPS_PROXY="http://10.41.13.6:3128"
ENV FTP_PROXY="http://10.41.13.6:3128"

RUN apt update

RUN apt -y upgrade

RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends git-core python3.9 python3-pip

RUN git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git

WORKDIR /cnlp_transformers

RUN git checkout --detach aa7410a596e8a0257cbfb51460594e7c3221e9c8

RUN pip3 install --upgrade pip

RUN pip3 install -e .

RUN pip3 install datasets

RUN pip3 install -U fastapi

COPY domain_rest.py /cnlp_transformers

COPY conf/ /cnlp_transformers/conf/

CMD uvicorn domain_rest:app --host 0.0.0.0
