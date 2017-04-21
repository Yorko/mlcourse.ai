import os
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=True, description='Run docker image locally')
    parser.add_argument("--docker_tag","-t", default='mlcourse_open', help='Docker image tag')
    args = parser.parse_args()


    os.system('docker run -it --rm -p 5022:22 -p 7777:7777 -v {0}:/notebooks -w /notebooks {1} jupyter'.format(os.getcwd(), args.docker_tag))

if __name__ == '__main__':
    main()


