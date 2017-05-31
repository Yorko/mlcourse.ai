#!/bin/bash -e

echo "Command: $1"

case $1 in
  shell )
    /bin/bash
    ;;
  h2o )
    /usr/local/h2o
    ;;
  jupyter )
    /usr/local/bin/jupyter notebook --ip="*" --no-browser --port 4545 --allow-root
    ;;
  zeppelin )
    ;;
  * )
    echo "Unknown command $1, starting shell"
    /bin/bash
    ;;
esac
