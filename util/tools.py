# encoding:utf-8
import yaml
from util.globalvars import GlobalVars
import os
import shutil
from pynvml import *
import requests
import json
from Crypto.Cipher import AES
import base64
import glob


# 检查并读取配置文件
def read_check_yaml(config_file):
    """
    :param config_file: 配置文件路径
    :return: True/False, dict/None
    """
    if not os.path.exists(config_file):
        return False, None
    config = read_yaml(config_file)
    if not check_config(config):
        return False, None
    return True, read_yaml(config_file)

# 检查配置文件的Key是否存在
def check_config(config):
    """
    :param config: config dict
    :return: Ture/False
    """
    is_normal = True
    for _ in ['ip', 'port', 'config_dir', 'windows_config_dir', 'log_file']:
        if _ not in config: is_normal=False
    return is_normal

# 读取yaml文件
def read_yaml(config_file):
    """
    :param config_file: 配置文件路径
    :return:  config dict
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except:
        with open(config_file, 'r', encoding='GB2312') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# 写入、覆盖配置文件
def write_yaml(config_file, yaml_config):
    """
    :param config_file: 配置文件路径
    :param yaml_config: 需要写入的dict
    :return: Null
    """
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f)

# 设置目录的全运行访问权限
def set_authority(file_dir):
    """
    :param file_dir: 文件目录
    :return: Null
    """
    os.chmod(file_dir, 0o777)

# 创建数据集对应的机型数据目录
def create_dataset_dir(supplier_list):
    """
    :param supplier_list: 机型类型（Birch、Rudolph、Kla......）
    :return:Null
    """
    # 创建数据集路径
    dataset_path = GlobalVars.get('datasets_path')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        set_authority(dataset_path)

    # 创建具体机型数据集路径
    for _supplier in supplier_list:
        _supplier_path = os.path.join(dataset_path, _supplier)
        if not os.path.exists(_supplier_path):
            os.mkdir(_supplier_path)
        set_authority(dataset_path)

    _temp_path = os.path.join(dataset_path, "temp")
    if not os.path.exists(_temp_path):
        os.mkdir(_temp_path)
    set_authority(dataset_path)

    # 删除多余的机型路径
    for f in os.listdir(dataset_path):
        if f == "temp" or f in supplier_list:
            continue
        shutil.rmtree(os.path.join(dataset_path, f))

# 创建临时目录下的类别目录
def create_temp_class_dir(defect_classes):
    """
    :param defect_classes: 缺陷种类
    :return: Null
    """
    # 创建temp目录
    dataset_path = GlobalVars.get('datasets_path')
    temp_path = os.path.join(dataset_path, "temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        set_authority(temp_path)

    # 创建缺陷类别目录
    defect_classes.append("Other")
    for _class in defect_classes:
        _temp_path = os.path.join(temp_path, _class)
        if not os.path.exists(_temp_path):
            os.mkdir(_temp_path)
        set_authority(temp_path)

    # 删除多余/遗留文件夹目录
    for f in os.listdir(temp_path):
        if f == "Other" or f in defect_classes:
            continue
        shutil.rmtree(os.path.join(temp_path, f))

def read_gpu_info():
    gpu_info = dict()
    # gpu_static = "GPU:"
    # nvmlInit()
    # deviceCount = nvmlDeviceGetCount()  # 显卡数量
    # for i in range(deviceCount):
    #     handle = nvmlDeviceGetHandleByIndex(i)
    #     info = nvmlDeviceGetMemoryInfo(handle)
    #     # gpu_name = nvmlDeviceGetName(handle).decode('utf-8')
    #     _total = round((info.total // 1048576) / 1024, 2)
    #     _used = round((info.used // 1048576) / 1024, 2)
    #     _free = round((info.free // 1048576) / 1024, 2)
    #     gpu_info[gpu_static + str(i)] = str(_used) + "_" + str(_total)
    gpu_info["GPU:0"] = str(1000) + "_" + str(1000)
        # temperature
        # nvmlDeviceGetTemperature(handle, 0)
    return gpu_info

# 加密、解密工具
def aes_encrypt(key, text):
    cipher = AES.new(key.encode('utf8'), AES.MODE_ECB)
    pad = lambda s: s+(AES.block_size - len(s) % AES.block_size) * chr(AES.block_size - len(s) % AES.block_size)
    text = pad(text)
    ciphertext = cipher.encrypt(text.encode('utf8'))
    return base64.b64encode(ciphertext)

def aes_decrypt(key, text):
    cipher = AES.new(key.encode('utf8'), AES.MODE_ECB)
    text = base64.b64decode(text)
    plaintext = cipher.decrypt(text)
    unpad = lambda s: s[0:-s[-1]]
    return unpad(plaintext).decode('utf8')


def get_project_path(project_name=None):
    """
        获取当前项目根路径
        :param project_name:
        :return: 根路径
    """
    p_name = 'maintain_tool' if project_name is None else project_name
    project_path = os.path.abspath(os.path.dirname(__file__))
    # Windows
    if project_path.find('\\') != -1: separator = '\\'
    # Mac、Linux、Unix
    if project_path.find('/') != -1: separator = '/'
    root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
    return root_path


image_extensions = [".jpg", ".jpeg", ".bmp", ".png"]
def get_img_list(folder_path):
    img_list = []
    for extension in image_extensions:
        img_list.extend(glob.glob(os.path.join(folder_path, "*{}".format(extension))))
    return img_list
