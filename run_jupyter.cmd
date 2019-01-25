

SET IMAGE="datamove/mlcourse_ai"

docker pull %IMAGE%

docker run --rm -v %cd%:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 %IMAGE% jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url="http://localhost:4545"


SET PWD=%cd%
SET USER=%username%
SET USER_ID=1000
SET GROUP_ID=1000
SET GROUP_NAME=fake

REM docker-compose -f docker\docker-compose.yaml up


