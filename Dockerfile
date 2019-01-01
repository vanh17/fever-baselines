FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir /fever/
RUN mkdir /fever/src
RUN mkdir /fever/config
RUN mkdir /fever/scripts
RUN mkdir /fever/data/
RUN mkdir /fever/data/fever/
RUN mkdir /fever/data/fever-data/
RUN mkdir /fever/data/models/


VOLUME /fever/

ADD requirements.txt /fever/
ADD gdown.pl /fever/
ADD src /fever/src/
ADD config /fever/config/
ADD scripts /fever/scripts/
ADD fever-data-small /fever/data/fever-data/
ADD fever-small /fever/data/fever/

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6

WORKDIR /fever/
RUN . activate fever
RUN apt-get install libffi-dev

RUN ["/bin/bash", "-c", "source activate fever && conda install pytorch torchvision -c pytorch"]
RUN ["/bin/bash", "-c", "source activate fever && pip install -r requirements.txt"]
RUN ["/bin/bash", "-c", "source activate fever && python src/scripts/prepare_nltk.py"]
