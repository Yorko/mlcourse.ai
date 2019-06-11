#!/bin/bash

# Usage:
# ./run_jupyter.sh [image]
#
# Parameters:
#   image - optional. A custom image to use instead of default. If it is given, 
#           we will not attemp to pull the latest image. This allows local image development.
#
# Container will be removed upon exit, but any jupyter settings and pakcages installed with
# pip install --user are kept in mlcourse.ai/home folder. 
# Remove them if you need a completely fresh image.
#
IMAGE=${1:-festline/mlcourse_ai}

#custom port
PORT=4545

# If we are called without a parameter to specify a new image, 
# let's make sure we are on the latest image
if [ $# -eq 0 ]; then
    echo "Will check for the latest image on the docker hub."
    docker pull $IMAGE
fi

exec docker run --rm -u $(id -u):$(id -g) -v "$PWD:/notebooks" -w /notebooks -e HOME=/notebooks/home -p $PORT:8888 $IMAGE jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url="http://localhost:$PORT"

# this allows for container to be created and persisted.
# which means that you can keep the changes you made, 
# i.e. if you installed more software with pip.
# 
# USER_ID=$(id -u) GROUP_ID=$(id -g) GROUP_NAME=$(id -gn) exec docker-compose -f docker/docker-compose.yaml up
