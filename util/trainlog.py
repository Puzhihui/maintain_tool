# encoding:utf-8
import time
# from model.vars import *
import os
from util.globalvars import GlobalVars

# 获取当前的log 文件名
def get_current_log_info():
  filename = time.strftime('%Y-%m-%d', time.localtime(time.time()))
  # train_log_path = TrainVars.get('logs_train_path') if TrainVars.get('logs_train_path') else GlobalVars.get('logs_train_path')
  train_log_path = GlobalVars.get('logs_train_path')
  file_path = os.path.join(train_log_path, filename + ".txt")
  current_time = time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime(time.time()))
  return file_path, current_time

# 写入message 到train log
def train_log_write(message):
  file_path, current_time = get_current_log_info()
  dir_path = os.path.dirname(file_path)
  os.makedirs(dir_path, exist_ok=True)
  with open(file_path, "a+", encoding="utf-8") as f:
    f.writelines(current_time + message + "\n")

# 获取某个起始时间到当前的log 内容
def get_train_log_content(str_start_datetime):
  content = ""
  start_timestamp = int(time.mktime(time.strptime(str_start_datetime, "%Y-%m-%d %H:%M:%S")))
  file_path, current_time = get_current_log_info()
  if not os.path.exists(file_path):
    return content

  with open(file_path, "r", encoding="utf-8") as f:
    print_info = f.readlines()

  index = -1
  for i, _print in enumerate(print_info):
    try:
      _str_print_date_time = _print.split(",")[0]
      _print_timestamp = int(time.mktime(time.strptime(_str_print_date_time, "%Y-%m-%d %H:%M:%S")))
      if _print_timestamp > start_timestamp:
        index = i
        break
    except:
      pass
  content = "".join(print_info[index:]) if index > -1 else ""
  content = content.rstrip("\n")

  return content


