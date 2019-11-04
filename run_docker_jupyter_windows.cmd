
REM
REM Attention: Windows Toolbox users - you MUST put the course repo in C:\Users\%username\mlcourse.ai
REM Attention: Use Ctrl-C to shut down. If you close the window, the container may still be running.
REM 

SET IMAGE="festline/mlcourse_ai"

REM check for new versin of the image
docker pull %IMAGE%

IF DEFINED  DOCKER_TOOLBOX_INSTALL_PATH (
FOR /F "tokens=* USEBACKQ" %%a in (`"docker-machine ip default"`) DO SET JUP_IP=%%a
)

IF DEFINED  DOCKER_TOOLBOX_INSTALL_PATH (

docker run -it -d --name mlcourse_ai --rm -u 1000:1000 -v /c/Users/%username%/mlcourse.ai:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 %IMAGE% jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url=http://%JUP_IP%:4545 & explorer "http://%JUP_IP%:4545" & docker attach mlcourse_ai

) ELSE (

docker run -it --rm -u 1000:1000 -v %cd%:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 -e JUPYTER_RUNTIME_DIR=/tmp -e JUPYTER_DATA_DIR=./ %IMAGE% jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url="http://localhost:4545"
)



