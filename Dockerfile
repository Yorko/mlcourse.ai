FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install --no-install-recommends -y apt-utils cmake git software-properties-common curl nano unzip openssh-server
RUN apt-get install -y python python-dev python-distribute python-pip

RUN pip install --upgrade pip
RUN pip install --upgrade numpy scipy matplotlib scikit-learn statsmodels pandas seaborn plotly jupyter
RUN pip install --upgrade watermark tqdm pydot

RUN apt-get -y install libboost-program-options-dev zlib1g-dev libboost-python-dev

# Vowpal Wabbit
RUN git clone git://github.com/JohnLangford/vowpal_wabbit.git && \
    cd vowpal_wabbit && make && make install
# python wrapper
RUN pip install --upgrade vowpalwabbit

# XGBoost
RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 && \
    cd python-package; python setup.py install && cd ../..

# LightGBM
RUN cd /usr/local/src && git clone --recursive --depth 1 https://github.com/Microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && cmake .. && make -j $(nproc) && \
    cd /usr/local/src/LightGBM/python-package && python setup.py install 

# TensorFlow 
RUN pip install --upgrade tensorflow  

# Keras with TensorFlow backend
RUN pip install --upgrade keras
