FROM continuumio/anaconda3:2019.03

# kaggle
RUN apt-get update  && apt-get install -y build-essential unzip && \
    conda update -y conda && conda update -y python && \
    pip install --upgrade pip && \
    apt-get -y install cmake

# mlcourse.ai
RUN apt-get install --no-install-recommends -y apt-utils curl nano 

# Kaggle
# Tensorflow doesn't support python 3.7 yet. See https://github.com/tensorflow/tensorflow/issues/20517
# Fix to install tf 1.10:: Downgrade python 3.7->3.6.6 and downgrade Pandas 0.23.3->0.23.2
RUN conda install -y python=3.6.6 
RUN conda upgrade -y notebook

# Kaggle
# The anaconda base image includes outdated versions of these packages. Update them to include the latest version.
RUN pip install --upgrade seaborn python-dateutil dask && \
    pip install pyyaml joblib pytagcloud husl geopy ml_metrics mne pyshp

# More upgrades to bring in sync with Kaggle
RUN pip install matplotlib==3.0.3 pandas==0.23.4 scikit-learn==0.21.3 numpy==1.17.0

# main python packages
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade numpy scipy matplotlib scikit-learn pandas seaborn plotly jupyter statsmodels
# bug with 5.7.0 jupyter/docker-stacks#721
# RUN pip3 install notebook==5.6.0
RUN pip install --upgrade tqdm pydot pydotplus watermark pillow mlxtend ua-parser user-agents plotly==4.1.0

# Graphviz, visualizing trees
RUN apt-get -y install graphviz 

#KAggle
# Make sure the dynamic linker finds the right libstdc++
ENV LD_LIBRARY_PATH=/opt/conda/lib


# kaggle
# beware that vw -V says 8.5.0 - @datamove
RUN cd /usr/local/src && \
    git clone -b 8.6.0 https://github.com/JohnLangford/vowpal_wabbit.git && \
    ./vowpal_wabbit/python/conda_install.sh && \
    # Reinstall in non-editable mode (without the -e flag)
    pip install vowpal_wabbit/python && \
    #above for some reasons do not install in common localtions - @datamove
    ln -s /usr/local/src/vowpal_wabbit/python/src/vowpalwabbit/vw /opt/conda/bin/vw

# Gradient boosting frameworks
RUN pip install -U xgboost lightgbm catboost


# PyTorch
# RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
# RUN pip3 install --upgrade torchvision

# TensorFlow 
# RUN pip3 install --upgrade tensorflow  

# Keras with TensorFlow backend
# RUN pip3 install --upgrade keras

# Facebook Prophet
RUN pip install --upgrade pystan cython

# Kaggle
# Latest version of fbprophet fails with incompatibility with PyStan. From the logs:
    # pystan:Something went wrong while unpickling the StanModel. Consider recompiling 
    # See: https://github.com/facebook/prophet/issues/775
RUN pip install fbprophet==0.4.post2

EXPOSE 8888
