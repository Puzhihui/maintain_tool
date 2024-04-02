# encoding:utf-8
import time
from models.vars import *
import os
from utils.globalvars import GlobalVars

# 获取当前的log 文件名
def get_current_log_info():
  filename = time.strftime('%Y-%m-%d', time.localtime(time.time()))
  train_log_path = TrainVars.get('logs_train_path') if TrainVars.get('logs_train_path') else GlobalVars.get('logs_train_path')
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
  # 转为时间戳
  #str_start_datetime = "2022-10-5 11:10:10"
  start_timestamp = int(time.mktime(time.strptime(str_start_datetime, "%Y-%m-%d %H:%M:%S")))

  file_path, current_time = get_current_log_info()
  content = "file is locked"
  if not os.path.exists(file_path):
    content = ""
    return content
  with open(file_path, "r", encoding="utf-8") as f:
    try:
      content = ""
      print_info = f.read().split("\n")
      for _print in print_info:
        try:
          _str_print_date_time = _print.split(": ")[0]
          _print_timestamp = int(time.mktime(time.strptime(_str_print_date_time, "%Y-%m-%d %H:%M:%S")))
          if _print_timestamp > start_timestamp:
            content += _print + "\n"
        except:
          pass
    except IOError:
      # file is locked
      pass
  return content[:-1]

