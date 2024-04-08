# encoding:utf-8
import threading
import json
import sys

import requests.exceptions, requests
from flask import Flask, request

from model.serverfunction import *
from util.deploylog import deploy_log_write
from model.vars import *

info_lock = threading.Lock() # 锁住线程，避免客户多线程同时请求ADCserver 冲突
app = Flask(__name__)

def record_request():
    requests_info = ServerVars.get('request_records')
    current_timestamp = datetime.datetime.now()
    current_datetime = current_timestamp.strftime("%m-%d %H")
    if current_datetime not in requests_info:
        requests_info[current_datetime] = 0
    requests_info[current_datetime] += 1
    ServerVars.set('request_records', requests_info)

def record_draw_request(port):
    testserverurl = "http://127.0.0.1:" + str(port)
    timeout_sec = 20
    while True:
        time.sleep(60 * 1)
        requests_info = ServerVars.get('request_records')
        last_record_timestamp = ServerVars.get('last_record_time')
        if time.time() - last_record_timestamp > 2 * 60:
            draw_request_info_images(requests_info)
            ServerVars.set('last_record_time', time.time())
            ServerVars.set('request_records', dict())
        # 判断服务是否退出
        try:
            response = requests.get(testserverurl, timeout= timeout_sec)
        except requests.exceptions.Timeout:
            break

# ---------------------deploy server flask server-----------------------#
def start_deploy_server(input_config = None):
    server_config(input_config)
    deploy_log_write("Server服务启动-开始")
    # 创建绘图子线程。
    job_record = threading.Thread(target=record_draw_request, args=(ServerVars.get('adc_server_port'),))
    job_record.start()

    # 开启flask
    app.run(host=input_config['ip'], port=ServerVars.get('adc_server_port'), debug=False, threaded=True)
    deploy_log_write("Server服务启动-结束")

# ---------------------init server start_deploy_server input var input_config-----------------------#
def structure_init_deploy_dict(global_var):
    init_server_config = dict()
    init_server_config['ip'] = global_var.get('ip')
    init_server_config['config_deploy_path'] = global_var.get('config_deploy_path')
    init_server_config['logs_deploy_path'] = global_var.get('logs_deploy_path')
    return init_server_config


# ---------------------first load run 模型加载-----------------------#
@app.before_first_request
def before_first_request():
    info_lock.acquire()
    init_model()
    info_lock.release()

# -------------before request 监控模型更新、重加载--------------#
@app.before_request
def before_request():
    info_lock.acquire()
    check_update_model()# 判断是否更新模型
    info_lock.release()

@app.route('/')
def ADCServerInfo():
    return 'ADC-Server Running!'


def exitprocess():
    deploy_log_write("server flask exit!")
    sys.exit()

# -------------模型推理--------------#
@app.route('/classify/', methods=['POST'])
def classify():
    _transforms = ServerVars.get('transform')
    _model = ServerVars.get('model')
    if request.method == 'POST':
        record_request()
        datas = request.get_json()
        device_info = device_dict_config(datas)
        try:
            results = predict(_model, device_info, _transforms)
        except Exception as e:
            deploy_log_write(str(e))
            exitprocess()

        response_info = json.dumps(results)
        del device_info, datas, results
        gc.collect()
        return response_info

