# encoding:utf-8
from flask import Flask, request
from util.globalvars import set_basic_config
import logging.handlers
import logging
app = Flask(__name__)

@app.route('/')
def ADCServerInfo():
    return 'ADC-1.0v'

def logging_init(log_file):
    loger = logging.getLogger()
    # 设置日志回滚，每天(24小时)回滚一次，最大备份15个文件，若是生成新的会将最早的一个文件删除
    trf = logging.handlers.TimedRotatingFileHandler(log_file, when='H', interval=24, backupCount=15)
    trf.suffix = "%Y-%m-%d_%H-%M-%S.log"
    #trf.suffix = "%Y-%m-%d.log" #备份日志文件格式
    # 设置最低打印日志等级
    trf.setLevel(logging.DEBUG)
    # 设置日志格式
    format = logging.Formatter('%(asctime)s - %(message)s')
    trf.setFormatter(format)
    loger.addHandler(trf)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename=log_file)


# ---------------------flask server-----------------------#
def start_server(param):
    logging_init(param['log_file'])
    set_basic_config(param)
    app.run(host=param['ip'], port=param['port'], debug=False, threaded=True)

# 接口函数
from interface.setAPI import imageformat_api, defectclass_api, datasetformat_api, supplier_api, dataset_path_api
from interface.basicAPI import gpuinfo_api
from interface.modeltrainAPI import adc_train_state_api, adc_train_api, train_print_api, train_config_api
from interface.modeldeployAPI import adc_server_api, adc_server_state_api, deploy_print_api, deploy_config_api

# set api
app.register_blueprint(imageformat_api)
app.register_blueprint(defectclass_api)
app.register_blueprint(datasetformat_api)
app.register_blueprint(supplier_api)
app.register_blueprint(dataset_path_api)

# basic api
app.register_blueprint(gpuinfo_api)

# train api
app.register_blueprint(adc_train_state_api)
app.register_blueprint(adc_train_api)
app.register_blueprint(train_print_api)
app.register_blueprint(deploy_config_api)

# server api
app.register_blueprint(adc_server_state_api)
app.register_blueprint(adc_server_api)
app.register_blueprint(deploy_print_api)
app.register_blueprint(train_config_api)


