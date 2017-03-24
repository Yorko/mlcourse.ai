# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM jupyter/scipy-notebook

MAINTAINER Yury Kashnitsky

USER root

RUN pip install --upgrade pip

RUN apt-get update && apt-get -y install vowpal-wabbit && apt-get -y install cmake

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
RUN wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl && \
    pip install tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl

# Keras with TensorFlow backend
RUN pip install keras

# Spark dependencies
# ENV APACHE_SPARK_VERSION 2.0.2
# RUN apt-get -y update && \
#    apt-get install -y --no-install-recommends openjdk-7-jre-headless && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
# RUN cd /tmp && \
#        wget -q http://d3kbcqa49mib13.cloudfront.net/spark-${APACHE_SPARK_VERSION}-bin-hadoop2.7.tgz && \
#        tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop2.7.tgz -C /usr/local && \
#        rm spark-${APACHE_SPARK_VERSION}-bin-hadoop2.7.tgz

# RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop2.7 spark

# Spark config
# ENV SPARK_HOME /usr/local/spark
# ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.3-src.zip
# ENV SPARK_OPTS --driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info

# update main conda packages
RUN conda update --quiet --yes numpy scipy pandas matplotlib seaborn statsmodels scikit-learn

# some other useful libraries
RUN pip install seaborn pydot plotly

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
