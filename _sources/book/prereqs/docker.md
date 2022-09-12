(prereq_docker)=

# Prerequisites

```{figure} /_static/img/ods_stickers.jpg
```

## Docker

_Note: the following instruction was used in 2019 during live course session, it might be a bit outdated._

All necessary software is already installed and distributed in the form of a [Docker container](https://cloud.docker.com/u/festline/repository/docker/festline/mlcourse_ai). Instructions:

### Docker on Linux and macOS
 - install [Docker](https://docs.docker.com/engine/installation/)
 - add your user to the docker group: `sudo usermod -aG docker your_user_name`
 - install git using your OS package manager
 - clone and download the [mlcourse.ai](https://github.com/Yorko/mlcourse.ai) repository
 - cd in the terminal into `mlcourse.ai`
 - execute `bash run_docker_jupyter.sh`. The first time it might take 5-10 minutes for image downloading
 - aim your browser to `localhost:4545`. You should see files from the mlcourse.ai folder
 - To test your setup, click on `docker_files` directory, open `check_docker.ipynb` and  execute all cells to make sure all the libraries are installed and work fine.

### Docker on Windows

If you meet the following requirements, install [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)

 - Windows 10 64bit: Pro, Enterprise, or Education (1607 Anniversary Update, Build 14393 or later).
 - Virtualization is enabled in BIOS. Typically, virtualization is enabled by default. This is different from having Hyper-V enabled. For    more detail see Virtualization must be enabled in Troubleshooting.
 - At least 4GB of RAM.

It's not the end of the world if you can't meet these requirements.
You can still use [Docker Toolbox](https://docs.docker.com/toolbox/overview) which is a good official alternative and with fewer requirements with to the Windows version. There are slight differences between Docker and Docker Toolbox for the end-user, but you can safely use both for now.

- When you run the installer, it may offer you to install git along. Mark a checkbox with this option if you don't have git on your system.
- In the case of Docker Toolbox, you may or may not need to delete your existing Virtualbox installation.
- Once the installation is complete, open docker (in case of docker toolbox open Docker CLI, it's called Docker Quickstart Terminal) and type: `> docker run hello-world`. It should run without errors.
- Open a Command-line terminal and clone the course repo: `git clone https://github.com/Yorko/mlcourse.ai`
- Warning for Docker Toolbox users: you must put your repo in your home dir, i.e. `C:\Users\%username%\mlcourse.ai`, otherwise the `run_docker_jupyter_windows.cmd` won't work. There is a workaround in case of a different location, but we don't assist with it.
- Change to mlcourse.ai directory: `cd mlcourse.ai` and run `run_docker_jupyter_windows.cmd`. Take a note on the local address the notebook reports, and aim your browser to this address. In the case of Windows 10 and Hyper-V, it should just be `http://localhost:4545`. In the case of Docker Toolbox, it's different. We implemented the autostart of your default browser with the correct address, but beware, that it may not work in Internet Explorer or Edge (for unknown reason). Use Firefox or Chrome then.
- In the browser, you should see the directory tree from your mlcourse.ai folder. Click on `docker_files`, open `check_docker.ipynb` and execute all cells to make sure all the libraries are installed and work fine.

### Docker tips
- Typically, Docker containers need a lot of disk space. The official mlcourse image requires some 6Gb of space.
- use `docker pull` to get new files from the repo to your locally downloaded repo.
- when you work with an assignment notebook, duplicate it first, and work with the duplicate. This way it's easier to pull changes to the repo since there will be no conflicts on the file level. If you'd like to work on a lecture notebook, do the same.
- You can install additional packages right in the Jupyter notebook with `pip install --user your_new_package`. They will be installed in mlcourse.ai `home` folder and will persist across Jupyter restart.
- optionally, you can modify the [docker_files/Dockerfile](https://github.com/Yorko/mlcourse.ai/blob/main/docker_files/Dockerfile) file, build a new image locally with `docker build -t <tag_name>`) and run `run_docker_jupyter.sh <tag_name>` (only supported under Linux/MacOS).
- Docker [documentation](https://docs.docker.com/engine/getstarted/) is full of concise and clear examples.

Few useful commands:

- `docker ps` – list all containers
- `docker stop $(docker ps -a -q)` – stop all containers
- `docker rm $(docker ps -a -q)` – remove all containers
- `docker images` - list all docker images
- `docker rmi <image_id>` – remove a docker image
