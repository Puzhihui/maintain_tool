# encoding:utf-8
# 创建ADC Train的相关进程、API。

from flask import Blueprint, request
import json
from datetime import datetime
from util.globalvars import GlobalVars
from util.config import read_config_centent, write_config_centent
import multiprocessing as mp
import time
from util.trainlog import get_train_log_content, train_log_write
from multiprocessing import freeze_support
from config.load_config import bat_cfg
from util.run_script_tools import run_bat, check_process_status, kill_process
freeze_support()  # 防止多开

train_process = mp.Process(target=run_bat)

adc_train_state_api = Blueprint('adc_train_state_api', __name__)
adc_train_api = Blueprint('adc_train_api', __name__)
train_print_api = Blueprint('train_print_api', __name__)
train_config_api = Blueprint('train_config_api', __name__)


# ADCTrainState
@adc_train_state_api.route("/ADCTrainState", methods=['POST'])
def ADC_train_state():
    global train_process
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            _isTrain = 1 if GlobalVars.get('train_state') else 0
            time.sleep(1)
            if _isTrain == 1 and check_process_status(train_process) is not True:
                _isTrain = 0
                GlobalVars.set('train_state', False)
                # 进程异常退出。
                train_log_write("train process exit!")
            results = {"ErrorCode": 0, "Msg": "Success", "Data": {"IsTrain":_isTrain}}
            return json.dumps(results)


# ADCTrain
@adc_train_api.route("/ADCTrain", methods=['POST'])
def ADC_train():
    global train_process
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        str_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        if request_param['Option'] == "start":
            bat_cfg.reload_yaml()  # 刷新配置
            train_args = bat_cfg.train_args
            train_args = [bat_cfg.client, train_args["train_model"], train_args["epoch"], train_args["batch_size"]]
            train_process = run_bat(bat_cfg.train_bat_path, args=train_args, create_console=True)
            GlobalVars.set('train_state', True)
            train_log_write("train process start!")
            results = {"ErrorCode": 0, "Msg": "Success", "Data": {"UpdateTime": str_time}}
        elif request_param['Option'] == "stop":
            kill_process(train_process)
            GlobalVars.set('train_state', False)
            train_log_write("train process exit!")
            results = {"ErrorCode": 0, "Msg": "Success", "Data": {"UpdateTime": str_time}}
        return json.dumps(results)

# TrainPrint
@train_print_api.route("/TrainPrint", methods=['POST'])
def train_print():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        now = datetime.now();str_time = now.strftime("%Y-%m-%d %H:%M")
        if request_param['Option'] == "get":
            _start_time = request_param['Datetime']
            data = get_train_log_content(_start_time)
            results = {"ErrorCode": 0, "Msg": "Success", "Data": data}
        return json.dumps(results)

# TrainConfig
@train_config_api.route("/TrainConfig", methods=['POST'])
def train_config():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            text = read_config_centent(GlobalVars.get('config_train_path'))
            results = {"ErrorCode": 0, "Msg": "Success", "Data": text}
        elif request_param['Option'] == "set":
            text = request_param['Data']
            write_config_centent(GlobalVars.get('config_train_path'), text)
            results = {"ErrorCode": 0, "Msg": "Success"}
        return json.dumps(results)
