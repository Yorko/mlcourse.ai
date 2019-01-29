

SET IMAGE="datamove/mlcourse_ai"

docker pull %IMAGE%

IF DEFINED  DOCKER_TOOLBOX_INSTALL_PATH (
docker run -it --rm -u 1000:1000 -v /c/users/%username%/mlcourse.ai:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 %IMAGE% jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url="http://localhost:4545"

) ELSE (

docker run -it --rm -u 1000:1000 -v %cd%:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 %IMAGE% jupyter-notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.password_required=False --NotebookApp.token='' --NotebookApp.custom_display_url="http://localhost:4545"
)

REM
REM Attention: Windows Toolbox users - you MUST put the course repo in C:\Users\%username\mlcourse.ai
REM Attention: Windows Toolbox users - check your URL in the docker controls.
REM Attention: Use Ctrl-C to shut down. If you close the window, the container may still run.
REM 

