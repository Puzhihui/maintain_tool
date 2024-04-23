# encoding:utf-8
import glob
import os.path
import random
import shutil

from flask import Blueprint, request
import json
from util.tools import read_yaml, write_yaml, create_dataset_dir, create_temp_class_dir, get_img_list
from util.globalvars import GlobalVars
from collections import defaultdict
from config.load_config import dataset_cfg

move_temp_img_api = Blueprint('move_temp_img_api', __name__)
imageformat_api = Blueprint('imageformat_api', __name__)
defectclass_api = Blueprint('defectclass_api', __name__)
datasetformat_api = Blueprint('datasetformat_api', __name__)
supplier_api = Blueprint('supplier_api', __name__)
dataset_path_api = Blueprint('dataset_path_api', __name__)


def temp_preprocess(img_list):
    temp_dict = defaultdict(lambda: defaultdict(list))
    for img in img_list:
        label = img.split(os.sep)[-2]
        recipe, img_name, supplier = img.split("@")[5], img.split("@")[9], img.split("@")[10]
        temp_dict[supplier][recipe][label].append(img)

    return temp_dict


def copy_imgs(img_list, save_path, is_remove=False):
    os.makedirs(save_path, exist_ok=True)
    for img in img_list:
        shutil.copy2(img, save_path)
        if is_remove:
            os.remove(img)


@move_temp_img_api.route("/AddToDataSet", methods=['POST'])
def move_temp_img():
    if request.method == 'POST':
        temp_path = os.path.join(GlobalVars.get('datasets_path'), "temp")
        img_list = get_img_list(os.path.join(temp_path, "*"))
        temp_dict = temp_preprocess(img_list)
        for supplier, supplier_dict in temp_dict.items():
            if supplier not in dataset_cfg.supplier2dataset:
                continue
            dataset = dataset_cfg.supplier2dataset[supplier]
            train_path = os.path.join(GlobalVars.get("datasets_path"), dataset["train"])
            val_path = os.path.join(GlobalVars.get("datasets_path"), dataset["val"])
            for recipe, recipe_dict in supplier_dict .items():
                for label, images in recipe_dict.items():
                    if label not in list(dataset["categories_{}".format(dataset_cfg.client)].keys()):
                        continue
                    random.shuffle(images)
                    val_num = int(dataset_cfg.val_ratio * len(images))
                    copy_imgs(images[:val_num], os.path.join(val_path, recipe, label), is_remove=True)
                    copy_imgs(images[val_num:], os.path.join(train_path, recipe, label), is_remove=True)

        img_list = get_img_list(os.path.join(temp_path, "*"))
        for img in img_list:
            os.remove(img)
    results = {"ErrorCode": 0, "Msg": "Success"}
    return json.dumps(results)


# image format
# 获取/设置图像保存格式, 比如：Area@Len@Name@OP-Label@Supplier@Device.jpg
@imageformat_api.route("/ImageFormat", methods=['POST'])
def image_format():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _image_format = global_config['ImageFormat']
            results = {"ErrorCode": 0, "Msg": "Success", "Data": _image_format}
        else:
            # set
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            set_format = request_param['ImageFormatList']
            global_config['ImageFormat'] = set_format
            write_yaml(GlobalVars.get('config_global_path'), global_config)
            results = {"ErrorCode": 0, "Msg": "Success"}
        return json.dumps(results)


# Defect Classes
# 获取/设置缺陷种类, 比如：['Surface', 'Good']
@defectclass_api.route("/DefectClassification", methods=['POST'])
def defect_class():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _defect_class = global_config['DefectClass']
            results = {"ErrorCode": 0, "Msg": "Success", "Data": _defect_class}
        else:
            # set
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _defect_class = request_param['DefectClass']
            global_config['DefectClass'] = _defect_class
            # create temp path
            create_temp_class_dir(_defect_class)                            # 创建缺陷路径
            write_yaml(GlobalVars.get('config_global_path'), global_config) # 写入全局配置文件
            results = {"ErrorCode": 0, "Msg": "Success"}
        return json.dumps(results)


# dataset format
# 获取/设置数据集形式,  IsDevice, 是否使用Deivce的形式进行存储数据. Key-DatasetFormat: -1:null;0:class;1:by device
@datasetformat_api.route("/DatasetFormat", methods=['POST'])
def dataset_format():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _IsDevice = global_config['IsDevice']
            # DatasetFormat: -1:null;0:class;1:by device
            results = {"ErrorCode": 0, "Msg": "Success", "Data": {"IsDevice": _IsDevice}}
        else:
            # set
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _IsDevice = request_param['IsDevice']
            global_config['IsDevice'] = _IsDevice
            write_yaml(GlobalVars.get('config_global_path'), global_config)
            results = {"ErrorCode": 0, "Msg": "Success"}
        return json.dumps(results)


# supplier format
# 获取/设置数据集存在的机型, 例如["Birch", "Rudolph", ......]
@supplier_api.route("/Supplier", methods=['POST'])
def supplier():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            _suppliers = global_config['Supplier']
            results = {"ErrorCode": 0, "Msg": "Success", "Data": _suppliers}
        else:
            # set
            global_config = read_yaml(GlobalVars.get('config_global_path'))
            supplier_list = request_param['SupplierList']
            global_config['Supplier'] = supplier_list
            create_dataset_dir(supplier_list)                               # 创建不同机型数据集目录, eg:dataset/Rudolph, dataset/Birch ...
            write_yaml(GlobalVars.get('config_global_path'), global_config) # 写入配置文件
            results = {"ErrorCode": 0, "Msg": "Success"}
        return json.dumps(results)


# dataset path
# 获取数据集共享目录
@dataset_path_api.route("/DataSetPath", methods=['POST'])
def dataset_path():
    if request.method == 'POST':
        request_param = request.get_json()
        results = dict()
        if request_param['Option'] == "get":
            dataset_share_path = GlobalVars.get('windows_datasets_path')
            results = {"ErrorCode": 0, "Msg": "Success", "Data": dataset_share_path}
        return json.dumps(results)