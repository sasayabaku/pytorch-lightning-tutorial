FROM nvidia/cuda:11.0-devel-ubuntu20.04

WORKDIR /code
ENV PYTHON_VERSION 3.7
ENV HOME /root
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

RUN apt-get -y update && \
    apt -y update

RUN apt -y install git make build-essential python3 python3-pip

RUN pip3 install torch torchvision 
RUN pip3 install pandas pytorch-lightning tensorboard jupyter