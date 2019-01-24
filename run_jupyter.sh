#!/bin/bash

# this allows for container to be created and persisted.
# which means that you can keep the changes you made, 
# i.e. if you installed more software with pip.
# 
USER_ID=$(id -u) GROUP_ID=$(id -g) GROUP_NAME=$(id -gn) exec docker-compose -f docker/docker-compose.yaml up
