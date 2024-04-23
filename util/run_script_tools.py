import subprocess
import time

import psutil
import requests
import json


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


def kill_process(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    parent.kill()


def check_server_monitor(monitor_url, timeout=10):
    isRun, pid = 1, None
    try:
        response = requests.post(monitor_url, timeout=timeout)
        results = response.text
        results = json.loads(results)
        if results["code"] != 0:
            isRun = 0
        pid = results["data"]
    except:
        isRun = 0
    return isRun, pid


def wait_server_start(monitor_url, timeout=60):
    start_time = time.monotonic()
    while time.monotonic() - start_time <= timeout:
        is_run, _ = check_server_monitor(monitor_url)
        if is_run:
            return True
    return False
