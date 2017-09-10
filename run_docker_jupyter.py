import os
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=True, description='Run docker image.')
    parser.add_argument("--docker_tag", "-t", default='festline/mlcourse_open', help='Docker image tag')
    parser.add_argument("--net_host", action='store_true', help='Whether to use --net=host with docker run (for Linux servers)')
    args = parser.parse_args()

    run_command = 'docker run -it {0} --rm -p 5022:22 -p 4545:4545 -v "{1}":/notebooks -w /notebooks {2} jupyter'.format(
        '--net=host' if args.net_host else '', os.getcwd(),  args.docker_tag)

    print('Running command\n' + run_command)
    os.system(run_command)

if __name__ == '__main__':
    main()




