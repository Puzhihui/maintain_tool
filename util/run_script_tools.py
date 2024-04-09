import subprocess
import psutil


def run_bat(bat_path, args=None, create_console=False):
    creationflags = subprocess.CREATE_NEW_CONSOLE if create_console else 0
    process_args = [bat_path] + args if args else [bat_path]
    process = subprocess.Popen(process_args, creationflags=creationflags)
    return process


def check_process_status(process):
    if process.poll() is None:
        return True
    else:
        return False


def kill_process(sub_process):
    # if check_process_status(sub_process):
    #     sub_process.kill()
    #     print('子进程已终止')

    parent = psutil.Process(sub_process.pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    parent.kill()


