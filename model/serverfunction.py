# encoding:utf-8
import csv

import matplotlib.pyplot as plt

from util.tools import read_yaml
from model.tools import *
from util.deploylog import deploy_log_write
from model.net.multitaskingmodel import MultitaskingModel
from model.torchdataloader import *

import os
import torch.nn as nn
from torchvision import datasets, transforms
import datetime
import time
import torch.nn.functional as F
import gc
from model.vars import *


# ---------------------初始化相关配置-----------------------#
def server_config(input_config):
    server_config = read_yaml(input_config['config_deploy_path'])
    init_config_var(server_config, input_config['logs_deploy_path'])
    deploy_log_write("init config end......")

# ---------------------初始化 ServerVars -----------------------#
def init_config_var(server_config, log_path):
    ServerVars.set('adc_server_port', server_config['adc_server_port'])
    ServerVars.set('cuda_device', server_config['cuda_device'])
    ServerVars.set('model_path', server_config['model_path'])
    ServerVars.set('model_name', server_config['model_name'])

    ServerVars.set('inference_batch_size', server_config['inference_batch_size'])
    ServerVars.set('num_workers', server_config['cuda_num_workers'])
    ServerVars.set('model_infer_config', server_config['model_infer_config'])

    # 特殊配置
    ServerVars.set('bad_prop_thresh', server_config['bad_prop_thresh'])
    ServerVars.set('isAreaControl', server_config['isAreaControl'])
    ServerVars.set('area_threshold', server_config['area_threshold'])
    ServerVars.set('limit_images', server_config['limit_images'])
    ServerVars.set('Illegal_keywords', server_config['Illegal_keywords'])


    # ServerVars.set('special_bad_prob', server_config['special_bad_prob'])
    # ServerVars.set('special_missing_bump', server_config['special_missing_bump'])
    # ServerVars.set('special_missing_bump_device', server_config['special_missing_bump_device'])
    # ServerVars.set('special_area_threshold', server_config['special_area_threshold'])
    # ServerVars.set('special_area_device', server_config['special_area_device'])

    # class list. 二分类 and 缺陷多分类
    if len(server_config['classes_txt']) == 1:
        ServerVars.set('binary_class', get_classes(server_config['classes_txt'][0]))
        ServerVars.set('multi_class', None)
    else:
        ServerVars.set('binary_class', get_classes(server_config['classes_txt'][0]))
        ServerVars.set('multi_class', get_classes(server_config['classes_txt'][1]))

    # transform set
    model_infer_configs = ServerVars.get('model_infer_config')
    infer_transform = []
    # 遍历图像的多种变化，送入模型推理
    for _transform in model_infer_configs:
        transforms_compose = []
        # 遍历每种变化的推理方式
        for _transforms_compose in _transform:
            if "CenterCrop" in _transforms_compose:
                _image_size = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.CenterCrop(_image_size))
            elif "Resize" in _transforms_compose:
                _image_size = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.Resize((_image_size, _image_size)))

            if "RandomAffine" in _transforms_compose:
                transforms_compose.append(transforms.RandomAffine(0, (0.08, 0.08)))
                transforms_compose.append(transforms.RandomHorizontalFlip(1))
        transforms_compose.append(Normalize())
        infer_transform.append(transforms.Compose(transforms_compose))

    ServerVars.set('transform', infer_transform)

    # model and config var
    ServerVars.set('model', None)
    ServerVars.set('model_pth_md5str', None)

    # log path
    ServerVars.set('logs_deploy_path', log_path)

    # adc request record
    ServerVars.set('request_records', dict())
    ServerVars.set('last_record_time', time.time())


# --------------------- init model load-----------------------#
def init_model():
    deploy_log_write("init model start......")
    torch.cuda.set_device(ServerVars.get('cuda_device')[0])
    class_nums = []
    class_nums.append(len(ServerVars.get('binary_class')))
    if ServerVars.get('multi_class') is not None:
        class_nums.append(len(ServerVars.get('multi_class')))
    ServerVars.set('class_nums', class_nums)

    # load model
    ServerVars.set('model', MultitaskingModel(model_name=ServerVars.get('model_name'), class_nums=class_nums))
    ServerVars.get('model').load_state_dict(torch.load(ServerVars.get('model_path'), map_location='cpu'))
    ServerVars.set('model', nn.DataParallel(ServerVars.get('model').to(torch.device("cuda:" + str(ServerVars.get('cuda_device')[0]))), device_ids=ServerVars.get('cuda_device')))
    ServerVars.get('model').eval()

    # 记录pth的md5 text
    ServerVars.set('model_pth_md5str', get_md5_str(ServerVars.get('model_path')))
    deploy_log_write("init model end......")

# --------------------- check model update-----------------------#
def check_update_model():
    current_pth_md5str = get_md5_str(ServerVars.get('model_path'))
    # model update if pth time change
    if ServerVars.get('model_pth_md5str') != current_pth_md5str:
        deploy_log_write("模型已更新，当前重加载，更新模型-开始")
        # load model
        ServerVars.set('model', MultitaskingModel(model_name=ServerVars.get('model_name'), class_nums=ServerVars.get('class_nums')))
        ServerVars.get('model').load_state_dict(torch.load(ServerVars.get('model_path'), map_location='cpu'))
        ServerVars.set('model', nn.DataParallel(
            ServerVars.get('model').to(torch.device("cuda:" + str(ServerVars.get('cuda_device')[0]))),
            device_ids=ServerVars.get('cuda_device')))
        ServerVars.get('model').eval()
        ServerVars.set('model_pth_md5str', current_pth_md5str)# 更新当前时间
        deploy_log_write("模型已更新，当前重加载，更新模型-结束")

# --------------------- request wafers ADC return device config-----------------------#
def device_dict_config(datas):
    device_info = dict()
    device_info['folder'] = datas['Folder']
    device_info['wafer_id'] = datas['WaferId']
    device_info['image_list'] = datas['ImagePathList']
    device_info['device'] = datas['RecipeName']
    device_info['areaList'] = datas['AreaList']

    device_info['bad_prop_thresh'] = float(ServerVars.get('bad_prop_thresh'))
    device_info['isAreaControl'] = ServerVars.get('isAreaControl')
    device_info['area_threshold'] = int(ServerVars.get('area_threshold'))
    device_info['inference_batch_size'] = int(ServerVars.get('inference_batch_size'))
    device_info['num_workers'] = int(ServerVars.get('num_workers'))
    device_info['limit_images'] = int(ServerVars.get('limit_images'))
    device_info['Illegal_keywords'] = ServerVars.get('Illegal_keywords')

    #class
    device_info['binary_class'] = ServerVars.get('binary_class')
    device_info['multi_class'] = ServerVars.get('multi_class')

    #cuda
    device_info['cuda_id'] = ServerVars.get('cuda_device')[0]
    return device_info

# --------------------- 获取软件发送的请求信息-----------------------#
def adc_request_info(data):
    dict_data = dict(data)
    dict_data['image_len'] = len(dict_data['ImagePathList'])
    del dict_data['ImagePathList']
    del dict_data['AreaList']
    return dict_data

# --------------------- 获取ADC结果的反馈信息-----------------------#
def adc_response_info(results):
    dict_results = dict(results)
    dict_results['data_len'] = len(dict_results['Data'])
    del dict_results['Data']
    return dict_results

# ---------------------ADC过算法-----------------------#
def predict(model, device_info, transforms):
    # 传入参数
    results = {"errorcode": str(0), "msg": "success"}

    img_source = device_info['image_list'] # list ['1.jpg', '2.jpg', ...]
    defect_areas = device_info['areaList'] # list [366.3, 2454.211 ......]
    device_id = device_info['device']
    device_dir = device_info['folder'] #存在漏检验证，不分Device path
    waferinfo = waferinfo_analysis(device_dir)

    num_worker = device_info['num_workers'] if len(img_source) > 150 else 0
    persistent_workers_flag = True if len(img_source) > 150 else False

    # bad threshold
    bad_prop_thresh = device_info['bad_prop_thresh']

    # area control set
    isAreaControl = device_info['isAreaControl']
    threshold_area = device_info['area_threshold']

    deploy_log_write(waferinfo + " "+ device_id + ". Nums: " + str(len(img_source)) + ". BadTH: " + \
                    str(bad_prop_thresh) + ". AreaCTL: " + str(isAreaControl) + ".AreaTH: " +str(threshold_area) + ".")

    # 检查device 是否正常
    is_normal, tmp_results = check_device_is_normal(device_info)
    if not is_normal:
        str_log = "result: " + tmp_results['msg'] + "."
        deploy_log_write(str_log)
        return tmp_results

    result = []
    # 记录模型推理的transform 多种方式
    transforms_num = len(transforms)
    deploy_dataset = DyployDataset(img_source, transforms, defect_areas)
    deploy_dataloader = DataLoader(deploy_dataset, batch_size=device_info['inference_batch_size'], num_workers=num_worker,
                                  pin_memory=True, drop_last=False, collate_fn=custom_collate_deploy, persistent_workers=persistent_workers_flag)

    bad_label = device_info['binary_class'][0] if device_info['binary_class'][0] != "Good" else device_info['binary_class'][1]
    good_label = device_info['binary_class'][1] if bad_label == device_info['binary_class'][0] else device_info['binary_class'][0]
    bad_index = 0 if device_info['binary_class'][0] != "Good" else 1

    # record normal or area control image nums
    model_normal_count = 0; area_control_count = 0

    # record Good/NG nums
    dict_result = dict()
    dict_result[bad_label] = 0;dict_result[good_label] = 0



    # record start time
    t1 = time.time()
    for images_batch, addresses, defect_areas in deploy_dataloader:
        if images_batch is None:
            continue
        _result = []
        with torch.no_grad():
            input_tensors = images_batch.to(torch.device("cuda:" + str(device_info['cuda_id'])))
            out_binary_pred, out_multi_pred = model(input_tensors)
            # torch.cuda.synchronize()
        p_binary = F.softmax(out_binary_pred, dim=1)
        pred_binary_numpy = p_binary.cpu().detach().numpy()

        # 缺陷多分类
        if device_info['multi_class'] is not None:
            p_multi = F.softmax(out_multi_pred, dim=1)
            p_multi_index = torch.argmax(p_multi, 1)
            pred_multi_numpy = p_multi_index.cpu().detach().numpy()

        # 遍历每一张图
        for i, temp_image in enumerate(addresses):
            bad_per_pred = 0.0
            for j in range(transforms_num):
                bad_per_pred = max(pred_binary_numpy[i + j * len(addresses)][bad_index], bad_per_pred)

            temp_result = dict()
            image_name = temp_image.replace('/', '\\')
            temp_result['image_name'] = image_name

            # area control / bad area value, defect with a default area of 0
            if defect_areas[i] > threshold_area and isAreaControl:
                temp_result['confidence'] = 0.9999
                temp_result['label'] = bad_label
                area_control_count += 1
            # model bad
            elif bad_per_pred > bad_prop_thresh:
                temp_result['confidence'] = str(round(bad_per_pred, 4))
                temp_result['label'] = bad_label
                model_normal_count += 1
                if device_info['multi_class'] is not None:
                    multi_class_index = pred_multi_numpy[i]
                    if bad_per_pred > 0.6 and p_multi[i][multi_class_index] > 0.8:
                        temp_result['label'] = device_info['multi_class'][multi_class_index]
            # model good
            else:
                temp_result['confidence'] = str(round(1 - bad_per_pred, 4))
                temp_result['label'] = good_label
                model_normal_count += 1
            _result.append(temp_result)

        for image_re in _result:
            if image_re['label'] == good_label:
                dict_result[good_label] += 1
            else:
                dict_result[bad_label] += 1
        for _ in _result: result.append(_)

    t2 = time.time()
    delta_time = (t2 - t1) / len(img_source) * 1000

    str_log = "result: " + good_label + "-" + str(dict_result[good_label]) + "; " + bad_label + "-" + str(dict_result[bad_label])
    str_log += "(normal:" + str(model_normal_count) + ";area:" + str(area_control_count) + ";)"
    str_log += "  avg runtime: " + str(round(delta_time, 3)) + 'ms/img.'
    deploy_log_write(str_log)

    # 删除/释放
    del deploy_dataset, deploy_dataloader, dict_result, str_log
    results['Data'] = result
    gc.collect()
    return results

def draw_request_info_images(request_records):
    deploy_log_temp_path = ServerVars.get('logs_deploy_path') + "Temp"
    record_temp_path = os.path.join(deploy_log_temp_path, "temp.csv")
    record_temp_image = os.path.join(deploy_log_temp_path, "DeployRequests.jpg")
    headers = ["Time", "Count"]
    max_count = 100
    count_hours = 48
    if os.path.exists(record_temp_path):
        with open(record_temp_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                key_time, value_count = row
                if key_time in request_records:
                    request_records[key_time] += int(value_count)
                else:
                    request_records[key_time] = int(value_count)
    # 画图
    plt.cla()
    plt.figure(figsize=(13, 7))
    ax = plt.gca()  # 获取边框
    ax.tick_params(axis='x', rotation = 45)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    display_key_times = []
    display_value_count = []
    for i in range(count_hours):
        _display_key = (datetime.datetime.now() + datetime.timedelta(hours=(-1 * (count_hours - 1 - i)))).strftime("%m-%d %H")
        display_key_times.append(_display_key)
        _temp_count_value = 0 if _display_key not in request_records else request_records[_display_key]
        display_value_count.append(_temp_count_value)

    plt.plot(display_key_times, display_value_count, label="Request Times")
    plt.legend()  # 显示图例
    # plt.xlabel('Date', fontsize=16)
    plt.ylabel('Number of requests', fontsize=16)
    plt.title(' ADC Running Info', fontsize=16)
    plt.savefig(record_temp_image)  # 图片保存
    plt.close()

    # save
    with open(record_temp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        # 倒叙写入记录最新
        request_records = dict(sorted(request_records.items(), key=lambda x:x[0], reverse=True))
        for index, (key_time, value_count) in enumerate(request_records.items()):
            if index == max_count:
                break
            writer.writerow([key_time, value_count])

