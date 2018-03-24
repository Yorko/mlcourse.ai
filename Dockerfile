FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install --no-install-recommends -y apt-utils software-properties-common curl nano unzip openssh-server
RUN apt-get install -y python3 python3-dev python-distribute python3-pip git

# main python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy scipy matplotlib scikit-learn pandas seaborn plotly jupyter statsmodels
RUN pip3 install --upgrade nose tqdm pydot pydotplus watermark geopy joblib

# Graphviz, visualizing trees
RUN apt-get -y install graphviz 

# Jupyter configs
RUN jupyter notebook --allow-root --generate-config -y
RUN echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# boost
RUN apt-get -y install libboost-program-options-dev zlib1g-dev libboost-python-dev

# JDK
RUN apt-get -y install openjdk-8-jdk
ENV CPLUS_INCLUDE_PATH=/usr/lib/jvm/java-8-openjdk-amd64/include/linux:/usr/lib/jvm/java-1.8.0-openjdk-amd64/include

# Vowpal Wabbit
RUN git clone git://github.com/JohnLangford/vowpal_wabbit.git && \
    cd vowpal_wabbit && make && make install
# python wrapper
RUN cd vowpal_wabbit/python && python3 setup.py install
RUN pip3 install --upgrade vowpalwabbit

# XGBoost
RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 

# xgboost python wrapper
RUN cd xgboost/python-package; python3 setup.py install && cd ../..

RUN apt-get -y install cmake 

# LightGBM
RUN cd /usr/local/src && git clone --recursive --depth 1 https://github.com/Microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && cmake .. && make -j $(nproc) 

# LightGBM python wrapper
RUN cd /usr/local/src/LightGBM/python-package && python3 setup.py install 

# CatBoost
RUN pip3 install --upgrade catboost

# PyTorch
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

# TensorFlow 
RUN pip3 install --upgrade tensorflow  

# Keras with TensorFlow backend
RUN pip3 install --upgrade keras

# Facebook Prophet
RUN pip3 install --upgrade pystan cython
RUN pip3 install --upgrade fbprophet


# == JAVA == (uncomment if you need it)
# Set locale to UTF-8
# Set limits
# Configure APT
# Install JDK
# Cleanup
# RUN locale-gen en_US.UTF-8 && echo LANG=\"en_US.UTF-8\" > /etc/default/locale && \
#    printf '%s\n%s\n%s\n%s\n' \
#        '* - memlock unlimited' \
#        '* - nofile 65536' \
#        '* - nproc 65536' \
#        '* - as unlimited' \
#        >> /etc/security/limits.conf && \
#    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | \
#        /usr/bin/debconf-set-selections && \
#    add-apt-repository -y ppa:webupd8team/java && \
#    apt-get -y update && \
#    apt-get install --no-install-recommends -y oracle-java8-installer oracle-java8-set-default && \
#    apt-get -y autoremove && apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/oracle-jdk8-installer

# == H2O == (uncomment if you need it)
# RUN cd /tmp && \
#    curl -L http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/4/h2o-3.10.4.4.zip \
#        > h2o-3.10.4.4.zip && \
#    unzip h2o-3.10.4.4.zip && \
#    mv ./h2o-3.10.4.4 /usr/local/ && \
#    rm h2o-3.10.4.4.zip && \
#    ln -s /usr/local/h2o-3.10.4.4 /usr/local/h2o

COPY docker_files/entry-point.sh /
# COPY docker_files/h2o /usr/local/bin/

# Final setup: directories, permissions, ssh login, symlinks, etc
RUN mkdir -p /home/user && \
    mkdir -p /var/run/sshd && \
    echo 'root:12345' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
  #  chmod a+x /usr/local/bin/h2o && \
    chmod a+x /entry-point.sh

WORKDIR /home/user
EXPOSE 22 4545

ENTRYPOINT ["/entry-point.sh"]
CMD ["shell"]
