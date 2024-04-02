# encoding:utf-8
# 创建ADC Train的相关进程、API。

from flask import Blueprint, request
import json
from datetime import datetime
from utils.globalvars import GlobalVars
from utils.config import read_config_centent, write_config_centent
import multiprocessing as mp
from models.train import start_train, structure_input_train_config
import time
from utils.trainlog import get_train_log_content, train_log_write
from multiprocessing import freeze_support
freeze_support()  # 防止多开

train_process = mp.Process(target=start_train)

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
            if _isTrain == 1 and train_process.is_alive() is not True:
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
            init_train_config_dict = structure_input_train_config(GlobalVars)
            train_process = mp.Process(target=start_train, args=(init_train_config_dict, ))  # server进程
            train_process.start()
            GlobalVars.set('train_state', True)
            results = {"ErrorCode": 0, "Msg": "Success", "Data": {"UpdateTime": str_time}}
        elif request_param['Option'] == "stop":
            GlobalVars.set('train_state', False)
            train_process.kill()
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