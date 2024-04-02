# encoding:utf-8
from datetime import datetime, timedelta
import datetime as dt
import os
import pymssql
import yaml

from utils.tools import read_yaml, set_authority
from models.tools import *
from utils.trainlog import train_log_write
from models.net.multitaskingmodel import MultitaskingModel
from models.torchdataloader import *
from models.vars import *

import torch.nn as nn
from torchvision import datasets, transforms
import time
from multiprocessing import Process, Manager
import cv2

# ---------------------1. 初始化train相关配置-----------------------#
def init_train_config(train_config):
    init_config_var(train_config)
    train_log_write("init config over......")

def init_config_var(config):
    train_config = read_yaml(config['config_train_path'])
    global_config = read_yaml(config['config_global_path'])
    datasets_path = config['datasets_path']  # dataset and temp
    logs_train_path = config['logs_train_path'] # 日志文件

    # train norm config
    TrainVars.set('iteration', train_config['iteration'])
    TrainVars.set('cuda_device', train_config['cuda_device'])
    TrainVars.set('model_name', train_config['model_name'])
    TrainVars.set('epochs', train_config['epochs'])
    TrainVars.set('model_path', train_config['model_path'])

    # class list. 二分类 and 缺陷多分类
    if len(train_config['classes_path']) == 1:
        TrainVars.set('binary_class', get_classes(train_config['classes_path'][0]))
        TrainVars.set('multi_class', None)
    else:
        TrainVars.set('binary_class', get_classes(train_config['classes_path'][0]))
        TrainVars.set('multi_class', get_classes(train_config['classes_path'][1]))

    # train config
    TrainVars.set('batch_size', train_config['batch_size'])
    TrainVars.set('model_train_transform', train_config['model_train_transform'])
    TrainVars.set('model_val_transform', train_config['model_val_transform'])
    TrainVars.set('acc_save_thresh', train_config['acc_save_thresh'])
    TrainVars.set('cuda_num_workers', train_config['cuda_num_workers'])
    TrainVars.set('device_max_thresh', train_config['device_max_thresh'])
    TrainVars.set('device_min_thresh', train_config['device_min_thresh'])
    TrainVars.set('train_val_rate', train_config['train_val_rate'])
    TrainVars.set('use_focalloss', train_config['use_focalloss'])
    TrainVars.set('lr', train_config['lr'])
    TrainVars.set('factor', train_config['factor'])
    TrainVars.set('patience', train_config['patience'])
    TrainVars.set('Illegal_keywords', train_config['Illegal_keywords'])

    # 特殊配置

    TrainVars.set('TempGoodDirModelFilt', train_config['TempGoodDirModelFilt'])
    TrainVars.set('IsClearDatasTrainEnd', train_config['IsClearDatasTrainEnd'])
    TrainVars.set('IsAutoAddDefect', train_config['IsAutoAddDefect'])
    TrainVars.set('IsStartAutoIter', train_config['IsStartAutoIter'])
    TrainVars.set('count_start_time', train_config['count_start_time'])
    TrainVars.set('interval_day', train_config['interval_day'])
    TrainVars.set('count_device_wafers_min', train_config['count_device_wafers_min'])
    TrainVars.set('device_overkill_threshold', train_config['device_overkill_threshold'])
    TrainVars.set('device_overkill_select_num', train_config['device_overkill_select_num'])
    TrainVars.set('device_underkill_threshold', train_config['device_underkill_threshold'])
    TrainVars.set('check_device_min', train_config['check_device_min'])
    TrainVars.set('new_device_image_num', train_config['new_device_image_num'])
    TrainVars.set('eagle_export_process', train_config['eagle_export_process'])
    TrainVars.set('eagle_host', train_config['eagle_host'])
    TrainVars.set('eagle_user', train_config['eagle_user'])
    TrainVars.set('eagle_password', train_config['eagle_password'])
    TrainVars.set('eagle_database', train_config['eagle_database'])

    # global set
    TrainVars.set('is_device', global_config['IsDevice'])
    TrainVars.set('defect_class', global_config['DefectClass'])
    TrainVars.set('image_format', global_config['ImageFormat'])
    TrainVars.set('suppliers', global_config['Supplier'])
    TrainVars.set('dataset_path', datasets_path)
    TrainVars.set('logs_train_path', logs_train_path)

# ---------------------2. 初始化模型-----------------------#
def init_model():
    cuda_device = TrainVars.get('cuda_device')
    torch.cuda.set_device(cuda_device[0])
    model_name = TrainVars.get('model_name')
    model_path = TrainVars.get('model_path')
    torch.cuda.set_device(TrainVars.get('cuda_device')[0])
    class_nums = []
    class_nums.append(len(TrainVars.get('binary_class')))
    if TrainVars.get('multi_class') is not None:
        class_nums.append(len(TrainVars.get('multi_class')))
    TrainVars.set('class_nums', class_nums)
    # load model
    model = MultitaskingModel(model_name=model_name, class_nums=class_nums)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = nn.DataParallel(model.cuda(), device_ids=cuda_device)
        model.eval()
    except Exception as e:
        train_log_write("模型初始化失败，详细：" + str(e))
        return False, None
    return True, model

# ---------------------3. 自定义特殊设置 -----------------------#
# 临时数据Good目录 过滤
def good_dir_filter(good_prob_thresh):
    log_result_str = ""
    isSuccess, model = init_model()
    if isSuccess:
        cuda_device = TrainVars.get('cuda_device')
        if str(TrainVars.get('binary_class')[0]).lower() == "good":
            good_index, label = 0, str(TrainVars.get('binary_class')[0])
        else:
            good_index, label = 1, str(TrainVars.get('binary_class')[1])
        temp_good_path = os.path.join(os.path.join(TrainVars.get('dataset_path'), "temp"), label)
        total_images = len(os.listdir(temp_good_path))
        if total_images == 0:
            log_result_str = "目录无缺陷图像！"
            return isSuccess, log_result_str
        delete_images = 0

        # transform set
        model_val_transform = TrainVars.get('model_val_transform')
        infer_transform = transforms_set(model_val_transform)
        temp_dataset = TmpGoodDataset(images_path=temp_good_path, data_transforms=infer_transform, good_label_idnex=good_index)
        temp_dataset_dataloader = DataLoader(temp_dataset, batch_size=16, num_workers=2,
                                      pin_memory=False, drop_last=False, collate_fn=custom_collate_temp)

        for images_batch, addresses in temp_dataset_dataloader:
            _result = []
            with torch.no_grad():
                input_tensors = images_batch.to(torch.device("cuda:" + str(cuda_device[0])))
                out_binary_pred, out_multi_pred = model(input_tensors)
            p = F.softmax(out_binary_pred, dim=1)
            pred = p.cpu().detach().numpy()
            for i, temp_image in enumerate(addresses):
                if pred[i][good_index] < good_prob_thresh:
                    try:
                        delete_images += 1
                        os.remove(temp_image)
                    except:
                        pass
        log_result_str = "当前清理Good数据为：" + str(delete_images) + "/" + str(total_images) + "。"
        for _ in range(6): torch.cuda.empty_cache()
    return isSuccess, log_result_str

# ---------------------清理图像中的非法关键词-----------------------#
# 临时数据过滤非法Device
def clear_tempdir_images():
    log_result_str = ""
    classes = TrainVars.get('binary_class')
    Illegal_keywords = TrainVars.get('Illegal_keywords')
    clear_nums = 0
    for _class in classes:
        temp_class_path = os.path.join(os.path.join(TrainVars.get('dataset_path'), "temp"), _class)
        if not os.path.exists(temp_class_path):
            log_result_str += _class + " defect dir not exists, please check!"
            continue
        images = os.listdir(temp_class_path)
        if len(images) == 0:
            continue
        for _image in images:
            _image_info = _image.split('@')
            if len(_image_info) == 10: #算法运维工具导出
                _device = _image_info[5]
                for _key in Illegal_keywords:
                    if _key in _device:
                        try:
                            os.remove(os.path.join(temp_class_path, _image))
                            clear_nums += 1
                        except:
                            continue
        log_result_str += _class + " defect dir clear image nums: " + str(clear_nums) + "."
        clear_nums = 0
    train_log_write(log_result_str)
# 执行SQL语句
def execute_data_Sql(sql, values=None):
    _host, _user = TrainVars.get('eagle_host'), TrainVars.get('eagle_user')
    _password, _database = TrainVars.get('eagle_password'), TrainVars.get('eagle_database')
    db = pymssql.connect(host=_host, user=_user, password=_password, database=_database, autocommit=True, charset='utf8')
    cursor = db.cursor()
    data = None
    if values is not None:
        cursor.execute(sql, values)
    else:
        cursor.execute(sql)
    data = cursor.fetchall()
    db.commit()
    db.close()
    return data

# 统计漏检、过检，新Device数据 导出数据库数据
def export_eagle_dataset():
    isSuccess = True
    # init need vars
    suppliers = TrainVars.get('suppliers') # 获取机型
    count_start_time = TrainVars.get('count_start_time') + " 00:00:00.000" # 统计开始时间
    count_end_time = datetime.now().strftime("%Y-%m-%d") + " 23:59:59.000" # 统计结束时间
    date_values = (count_start_time, count_end_time)
    without_count_dw_nums = TrainVars.get('count_device_wafers_min')
    Illegal_keywords = TrainVars.get('Illegal_keywords')

    # 按照数据源机型遍历数据库
    for _supplier in suppliers:
        # 取出统计的Wafer id
        sql_query_wafer_start = "SELECT WaferId, RecipeName, LotId, WaferName FROM ReviewTask where WaferId in (SELECT Id from Wafer where WaferTime > %s and WaferTime < %s and AiPostState <> 0 and Supplier like '"
        # TaskState = 5 表示OP复判完成的片子
        sql_query_wafer_end = "%') and TaskState = 5"
        data = None
        while True:
            try:
                data = execute_data_Sql(sql_query_wafer_start + _supplier + sql_query_wafer_end, date_values)
                break
            except Exception as e:
                train_log_write("网络不稳定，数据库连接失败，等待10秒重新尝试，错误详细：" + str(e))
                time.sleep(10)
        train_log_write("统计晶圆数量：" + str(len(data)) + "/" + _supplier)

        # 统计少于N片的wafer 统计 without_count_devices
        # 统计新增的device
        _supplier_path = os.path.join(TrainVars.get('dataset_path'), _supplier)
        devices_list = os.listdir(_supplier_path)
        ad_device = []# 记录需要添加的新Device
        waferid_device = dict() # dict{wafer id: device_name}
        waferid_basic_info = dict() #dict{wafer id: [lotid, wafername]}
        without_count_devices = [] # 晶圆量不足，不需要统计的device 数组
        devices_wafers_count = dict() # dict{device: wafer nums}
        for index, (wafer_id, device, lotid, wafername) in enumerate(data):
            # 检查非法字符
            if len(Illegal_keywords) != 0:
                for _Illegal_keyword in Illegal_keywords:
                    if _Illegal_keyword in device.lower():
                        continue
            waferid_device[wafer_id] = device
            waferid_basic_info[wafer_id] = [lotid, wafername]
            devices_wafers_count[device] = devices_wafers_count[device] + 1 if device in devices_wafers_count.keys() else 1
        for device in devices_wafers_count.keys():
            if devices_wafers_count[device] < without_count_dw_nums:
                without_count_devices.append(device)
            elif device not in devices_list: #需要添加的devices，情况1:device 不在数据集中
                ad_device.append(device)
            elif len(os.listdir(os.path.join(_supplier_path, TrainVars.get('defect_class')[0]))) < TrainVars.get('check_device_min') or \
                    len(os.listdir(os.path.join(_supplier_path, TrainVars.get('defect_class')[1]))) < TrainVars.get('check_device_min'):
                ad_device.append(device)#需要添加的devices，情况2: device中的类别缺陷图像少于55张，需要填充

        # 遍历Defect表，取Image数据
        query_wafers_num = 3;count_num = query_wafers_num;temp_waferids = [];count_waferid = 0
        defects_info = [] # 记录defect 漏检/过检 缺陷图像信息
        wafers_num = len(waferid_device)
        eagle_defect_start_time = time.time()
        count_defect_sql_start = "select WaferId, ReviewImagePath, BincodeReviewed, BincodeAI, Length, Width, Area from Defect WITH(NOLOCK) where WaferId in ("
        count_defect_sql_end = ") and BincodeAI is not null and BincodeReviewed is not null and ((BincodeReviewed = 'Good' and BincodeAI <> 'Good') or  (BincodeReviewed <> 'Good' and BincodeAI = 'Good'))"
        query_fail_waferids = []
        for wafer_id, device_name in waferid_device.items():
            count_waferid += 1  # 计数
            if device_name in without_count_devices:
                continue
            temp_waferids.append(wafer_id)  # 记录wafer id 临时数组
            query_wafers_num -= 1 # 计数
            # 三片晶圆的wafer id 数组查询一次数据库，获取Defect 列表
            if query_wafers_num == 0 or count_waferid == wafers_num:
                waferid_str = ",".join(str(_wafer_id) for _wafer_id in temp_waferids)
                value_waferid_str = r"%s" % waferid_str # "1,2,3"
                try:
                    data = execute_data_Sql(count_defect_sql_start + value_waferid_str + count_defect_sql_end)
                    query_wafers_num = count_num
                    for _ in data: defects_info.append(_)
                    temp_waferids.clear()
                    time.sleep(0.0001)
                except Exception as e:
                    for _ in temp_waferids: query_fail_waferids.append(_) # 记录统计失败的wafer id
                    temp_waferids.clear()
                    time.sleep(0.01)

        # error count wafer id recount
        errors_wafers_num = len(query_fail_waferids)
        errors_count = 0 # 如果继续查询失败，说明该数据库查询存在问题
        if errors_wafers_num != 0:
            for _waferid in query_fail_waferids:
                try:
                    data = execute_data_Sql(count_defect_sql_start + str(_waferid) + count_defect_sql_end)
                    for _ in data: defects_info.append(_)
                except:
                    errors_count += 1
                    time.sleep(0.01)

        # 未统计的wafer id较多，则退出
        if wafers_num != 0 and errors_count/wafers_num > 0.001:
            isSuccess = False
            break

        # defects_info 为漏检、过检数据, 按照Device统计 漏检、过检数量
        underkill_dict = dict() # {device:[defect index 1, defect index 2, ......]}
        overkill_dict = dict() # {device:[defect index 1, defect index 2, ......]}  defects_info 索引
        for index, row in enumerate(defects_info):
            _device = waferid_device[row[0]]
            _bincode_review = row[2]
            if _bincode_review == "Good":
                if _device not in overkill_dict:
                    overkill_dict[_device] = []
                    overkill_dict[_device].append(index)
                else:
                    overkill_dict[_device].append(index)
            else:
                if _device not in underkill_dict:
                    underkill_dict[_device] = []
                    underkill_dict[_device].append(index)
                else:
                    underkill_dict[_device].append(index)

        # Eagle 漏/过检 数据转移至本地temp文件夹 ......
        transfers_datas = []
        device_underkill_threshold = TrainVars.get('device_underkill_threshold')
        device_overkill_threshold = TrainVars.get('device_overkill_threshold')
        bad_label = TrainVars.get('defect_class')[0] if TrainVars.get('defect_class')[0] != "Good" else TrainVars.get('defect_class')[1]
        good_label = TrainVars.get('defect_class')[0] if TrainVars.get('defect_class')[0] != bad_label else TrainVars.get('defect_class')[1]
        temp_bad_save_path = os.path.join(os.path.join(TrainVars.get('dataset_path'), "temp"), bad_label)
        temp_good_save_path = os.path.join(os.path.join(TrainVars.get('dataset_path'), "temp"), good_label)
        # underkill dir data transfer
        for device, image_ids in underkill_dict.items():
            if len(image_ids) < device_underkill_threshold:
                continue
            for _image_id in image_ids:
                # defects_info[_image_id].append(device) # image 增加device name
                wafer_id, image_path, op_review, bincode_ai, length, width, area = defects_info[_image_id][:]
                try:
                    width = round(float(width), 2)
                    length = round(float(length), 2)
                except:
                    width = length = ""
                lotid, wafername = waferid_basic_info[wafer_id][:]
                _name = os.path.basename(image_path).split(".")[0]
                try:
                    area = str(area).split(".")[0]
                except:
                    area = ""
                bincode_ai = str(bincode_ai);op_review = str(op_review)
                export_image_name = "@".join(str(_item) for _item in [area, length, width, op_review, bincode_ai, device,\
                                                                      lotid, wafername, _name, _supplier])
                export_image_name = r"%s" % export_image_name + ".jpg"
                transfers_datas.append([image_path, os.path.join(temp_bad_save_path, export_image_name)])
        # overkill dir data transfer
        for device, image_ids in overkill_dict.items():
            if len(image_ids) < device_overkill_threshold:
                continue
            random.shuffle(image_ids)
            for _image_id in image_ids[:TrainVars.get('device_overkill_select_num')]:
                # defects_info[_image_id].append(device)  # image 增加device name
                wafer_id, image_path, op_review, bincode_ai, length, width, area = defects_info[_image_id][:]
                try:
                    width = round(float(width), 2)
                    length = round(float(length), 2)
                except:
                    width = length = ""
                lotid, wafername = waferid_basic_info[wafer_id][:]
                _name = os.path.basename(image_path).split(".")[0]
                try:
                    area = str(area).split(".")[0]
                except:
                    area = ""
                bincode_ai = str(bincode_ai)
                op_review = str(op_review)
                export_image_name = "@".join(
                    str(_item) for _item in [area, length, width, op_review, bincode_ai, device, \
                                             lotid, wafername, _name, _supplier])
                export_image_name = r"%s" % export_image_name + ".jpg"
                transfers_datas.append([image_path, os.path.join(temp_good_save_path, export_image_name)])

        # new device data transfer（Eagle Dataset export）
        delta = timedelta(days=-60) # 往前两个月的时间
        shifted_date = dt.date.today() + delta
        _temp_start_time = shifted_date.strftime("%Y-%m-%d") + " 00:00:00.000"
        new_device_query_time = (_temp_start_time, count_end_time,) # 今天为统计结束时间
        train_log_write("新Device 数量：" + str(len(ad_device)) + ".")
        data = execute_data_Sql(sql_query_wafer_start + _supplier + sql_query_wafer_end, new_device_query_time)
        device_waferid = dict()
        # device:[wafer id1, wafer id2......]
        for index, (wafer_id, device) in enumerate(data):
            if device not in device_waferid:
                device_waferid[device] = []
            device_waferid[device].append(wafer_id)

        for _device in ad_device:
            if _device not in device_waferid:
                continue
            _waferids = device_waferid[_device]
            defect_datas = []
            for _wafer_id in _waferids:
                _sql = "SELECT WaferId, Area, ReviewImagePath, BincodeReviewed, BincodeAI, ConfidenceAI, Length, Width FROM Defect where WaferId = " + str(_wafer_id)
                _sql += " and BincodeReviewed is not null and def.ReviewImagePath is not null"
                data = execute_data_Sql(_sql)
                for _ in data: defect_datas.append(_)
            random.shuffle(defect_datas)
            _good_num = 0
            _bad_num = 0
            dest_image_num = TrainVars.get('new_device_image_num')
            for row in defect_datas:
                wafer_id = row[0]
                try:
                    area = str(row[1]).split(".")[0]
                except:
                    area = ""
                image_path = row[2]
                op_review = row[3]
                bincode_ai = row[4]
                try:
                    width = round(float(row[-1]), 2)
                    length = round(float(row[-2]), 2)
                except:
                    width = length = ""

                _name = os.path.basename(image_path).split(".")[0]
                lotid, wafername = waferid_basic_info[wafer_id][:]
                bincode_ai = str(bincode_ai)
                op_review = str(op_review)
                export_image_name = "@".join(str(_item) for _item in [area, length, width, op_review, bincode_ai, _device, lotid, wafername, _name, _supplier])
                export_image_name = r"%s" % export_image_name + ".jpg"
                if "Good" == op_review and _good_num < dest_image_num:
                    transfers_datas.append([image_path, os.path.join(temp_good_save_path, export_image_name)])
                    _good_num += 1
                elif "Good" != op_review and _bad_num < dest_image_num:
                    transfers_datas.append([image_path, os.path.join(temp_bad_save_path, export_image_name)])
                    _bad_num += 1
                if _good_num >= dest_image_num and _bad_num >= dest_image_num:
                    break

        # 转移图像
        train_log_write(_supplier + " 开始导出数据......")
        with Manager() as manager:
            transfer_images_queue = manager.Queue()
            [transfer_images_queue.put(_) for _ in transfers_datas]
            numProcess = 16
            [transfer_images_queue.put(None) for _ in range(numProcess)]
            pList = []
            for _ in range(numProcess):
                p = Process(target=transfer_images, args=(transfer_images_queue,))
                p.start()
                pList.append(p)
            for p in pList:
                p.join()
        train_log_write(_supplier + " 导出数据结束......")
    return isSuccess

# 获取eagle数据库的defect列表
def get_defect_info(waferid_device):
    count_num = 3
    wafers_num = len(waferid_device)
    num = count_num
    lst = []
    count = 0
    result_info = []
    result_total = []
    t1 = time.time()
    count_images_sql_start = "select WaferId, ReviewImagePath, BincodeReviewed, BincodeAI, ConfidenceAI, BincodeFinal, Area from Defect WITH(NOLOCK) where WaferId in ("
    count_images_sql_total_end = ") and BincodeAI is not null"
    error_waferids = []
    for wafer_id, device_name in waferid_device.items():
        lst.append(wafer_id)
        num -= 1
        count += 1

        if num == 0 or count == wafers_num:
            inList = ",".join(str(i) for i in lst);value_str = r"%s" % inList
            try:
                data = execute_data_Sql(count_images_sql_start + value_str + count_images_sql_total_end)
                num = count_num
                for _ in data:result_info.append(_)
                lst.clear();value_str = ""
            except:
                for _ in lst:error_waferids.append(_)
                lst.clear();value_str = ""
                time.sleep(0.01)
    # error count
    wafers_num = len(error_waferids)
    count = 0
    for _waferid in error_waferids:
        lst.append(_waferid)
        num -= 1
        count += 1
        if num == 0 or count == wafers_num:
            inList = ",".join(str(i) for i in lst);value_str = r"%s" % inList
            try:
                data = execute_data_Sql(count_images_sql_start + value_str + count_images_sql_total_end)
                num = count_num
                for _ in data:result_info.append(_)
                lst.clear();value_str = ""
            except:
                # print(lst)
                lst.clear();value_str = ""
                time.sleep(0.01)

    runtime = str(round(time.time() - t1, 2))
    return result_info

# 清理数据集
def clear_dataset():
    # init var
    suppliers_path = TrainVars.get('dataset_path')
    isClear, save_images_num = TrainVars.get('IsClearDatasTrainEnd')
    class_path = TrainVars.get('class_path')
    binary_class = TrainVars.get('binary_class')
    if not isClear:
        # 写入 return
        return

    suppliers = os.listdir(suppliers_path)
    delete_path = []
    for _supplier in suppliers:
        if _supplier == "temp":
            continue
        device_path = os.path.join(suppliers_path, _supplier)
        devices = os.listdir(device_path)
        clear_device_num = 0
        # 遍历device
        for _device in devices:
            _path = os.path.join(device_path, _device)
            good_path = os.path.join(_path, binary_class[1])
            bad_path = os.path.join(_path, binary_class[0])
            _clear_path = [good_path, bad_path]
            _isClear = False

            # clear Good/Bad dir
            for _dest_path in _clear_path:
                if os.path.exists(_dest_path):
                    images = os.listdir(_dest_path)
                    # dir data greater than save_image_num
                    if len(images) > save_images_num:
                        _isClear = True
                        random.shuffle(images)
                        for _ in images[save_images_num:]:
                            delete_path.append(os.path.join(_dest_path, _))
            if _isClear:
                clear_device_num += 1
    with Manager() as manager:
        delete_images_queue = manager.Queue()
        [delete_images_queue.put(_) for _ in delete_path]
        numProcess = 16
        [delete_images_queue.put(None) for _ in range(numProcess)]
        pList = []
        for _ in range(numProcess):
            p = Process(target=delete_image, args=(delete_images_queue,))
            p.start()
            pList.append(p)
        for p in pList:
            p.join()


# 自动增加缺陷
def auto_add_datadefect():
    suppliers_path = TrainVars.get('dataset_path')
    isAdd, dest_defect_num, defect_path = TrainVars.get('IsAutoAddDefect')
    binary_class = TrainVars.get('binary_class')
    good_label = binary_class[0] if binary_class[0] == "Good" else binary_class[1]
    bad_label = binary_class[1] if binary_class[0] == good_label else binary_class[0]
    if not isAdd:
        return
    defect_images = os.listdir(defect_path)
    defect_len = len(defect_images)
    suppliers = os.listdir(suppliers_path)
    for _supplier in suppliers:
        if _supplier == "temp":
            continue
        _supplier_path = os.path.join(suppliers_path, _supplier)
        devices = os.listdir(_supplier_path)
        for _device in devices:
            _device_good_path = os.path.join(os.path.join(_supplier_path, _device), good_label)
            _device_bad_path = os.path.join(os.path.join(_supplier_path, _device), bad_label)
            if not os.path.exists(_device_good_path) or not os.path.exists(_device_bad_path):
                continue
            _good_imaegs = os.listdir(_device_good_path)
            _good_len = len(_good_imaegs)
            _bad_len = len(os.listdir(_device_bad_path))
            if _good_len < 300 or _bad_len >= dest_defect_num:
                continue
            _fake_num = len(os.listdir(_device_bad_path))
            add_image_num = dest_defect_num - _bad_len
            for _ in range(add_image_num):
                _defect_id = random.randint(0, defect_len - 1)
                _defect_image_path = os.path.join(defect_path, defect_images[_defect_id])
                _good_image_id = random.randint(0, _good_len - 1)
                _good_image_path = os.path.join(_device_good_path, _good_imaegs[_good_image_id])
                _good_image = cv2.imread(_good_image_path)
                _defect_image = cv2.imread(_defect_image_path)
                new_image = combine_image(_good_image, _defect_image)
                cv2.imwrite(os.path.join(_device_bad_path, "fake-" + str(_ + _fake_num) + ".jpg"), new_image)

# 训练结束后，将temp文件数据转移至Datasets
def transfer_online_train_data():
    temp_path = os.path.join(TrainVars.get('dataset_path'), "temp")
    dataset_is_device = TrainVars.get('is_device')
    defect_classes = TrainVars.get('defect_class')
    image_format = TrainVars.get('image_format')
    suppliers = TrainVars.get('suppliers')
    datasets_path = TrainVars.get('dataset_path')
    binary_class = TrainVars.get('binary_class')

    trans_data = []
    for _class in binary_class:
        _temp_class_path = os.path.join(temp_path, _class)
        images = os.listdir(_temp_class_path)
        for _image in images:
            _image_path = os.path.join(_temp_class_path, _image)
            _suplies = _image.split("@")[-1].split(".")[0]
            _image_device = _image.split("@")[5]
            _save_device_path = os.path.join(datasets_path, _suplies)
            _save_device_path = os.path.join(_save_device_path, _image_device)

            _save_device_class_path = os.path.join(_save_device_path, _class)
            if not os.path.exists(_save_device_path):
                os.mkdir(_save_device_path)
                set_authority(_save_device_path)
            if not os.path.exists(_save_device_class_path):
                os.mkdir(_save_device_class_path)
                set_authority(_save_device_class_path)
            trans_data.append([_image_path, os.path.join(_save_device_class_path, _image)])
    numProcess = 8
    with Manager() as manager:
        image_names = manager.Queue()
        for name in trans_data: image_names.put(name)
        [image_names.put(None) for _ in range(numProcess)]
        pList = []
        for _ in range(numProcess):
            p = Process(target=trans_device_image, args=(image_names,))
            p.start()
            pList.append(p)
        for p in pList:
            p.join()
    transfer_nums = len(trans_data)
    train_log_write("数据转移：" + str(transfer_nums))
    return transfer_nums

def temp_datas_filter():
    isFilter, good_prob_thresh = TrainVars.get('TempGoodDirModelFilt')
    if isFilter:
        train_log_write("临时数据-Good过滤-开始")
        try:
            isSuccess, log_result_str = good_dir_filter(good_prob_thresh)
            if not isSuccess:
                train_log_write("临时数据-Good过滤-错误，算法模型初始化失败。")
                return False
        except Exception as e:
            train_log_write("临时数据-Good过滤-错误：" + str(e))
            return False
        train_log_write("临时数据-Good过滤-结束, " + log_result_str)

# ---------------------4. 训练任务 -----------------------#
def train_task():
    train_log_write("训练启动-开始")
    # tip: 清理非法device，删除临时数据中的非法关键词
    train_log_write("临时数据非法Device清理-开始")
    clear_tempdir_images()
    train_log_write("临时数据非法Device清理-结束")

    # task 1: 判断是否过滤temp中的Good数据
    temp_datas_filter()

    # 训练模型，自动/手动迭代开始
    N = 0
    while N < TrainVars.get('iteration'):
        # task 2: 初始化模型
        train_log_write("算法模型初始化-开始。")
        isSuccess, model = init_model()
        if not isSuccess:
            train_log_write("算法模型初始化-失败。")
            return False
        train_log_write("算法模型初始化-结束。")

        # task 3: 自动迭代处理
        isStartAutoIter = TrainVars.get('IsStartAutoIter')
        if isStartAutoIter:
            train_log_write("全自动迭代-开始。")
            while True:
                train_log_write("全自动迭代模式进行中，导出Eagle数据")
                if interval_day(TrainVars.get('count_start_time')) >= TrainVars.get('interval_day'):
                    # try:
                    isSuccess = export_eagle_dataset()
                    if isSuccess:
                        temp_datas_filter()
                        break #正常情况下会 直接导出数据，然后break
                    # 可能是数据库链接失败，有波动，不稳定；需要打印错误，并且休眠之后再做尝试
                    train_log_write("数据库链接失败，请查看数据库链接是否正常.")
                time.sleep(60 * 60 * 1)  #1小时
            train_log_write("全自动迭代-结束。")

        # task 4: 训练
        train_log_write("训练数据-开始")
        try:
            train_sucess_flag, train_devices = train(model)
            if not train_sucess_flag:
                train_log_write("训练异常，结束训练")
                break
        except Exception as e:
            train_log_write(str(e))
            return
        train_log_write("训练数据-结束")

        # 特殊工作
        # task 5: 转移数据
        train_log_write("临时数据转移数据集-开始")
        try:
            transfer_nums = transfer_online_train_data()
        except Exception as e:
            train_log_write(str(e))
            return
        train_log_write("临时数据转移数据集-结束")


        # 4. clear dataset
        train_log_write("清理冗余数据集-开始")
        try:
            clear_dataset()
        except Exception as e:
            train_log_write(str(e))
            return
        train_log_write("清理冗余数据集-结束")

        # 5. 自动增加bad缺陷
        train_log_write("自动增加缺陷-开始")
        try:
            auto_add_datadefect()
        except Exception as e:
            train_log_write(str(e))
            return
        train_log_write("自动增加缺陷-结束")

        N += 1
        # 释放显存
        imshow_str = '第' + str(N) + '次模型迭代结束！'
        for _ in range(6): torch.cuda.empty_cache()
        train_log_write(imshow_str + " 当前已释放显存。")
    train_log_write("训练启动-结束")

def transforms_set(transforms_config, isTrain = False):
    re_transforms = []
    # 遍历图像的多种变化，送入模型推理
    for _transform in transforms_config:
        transforms_compose = []
        # 训练过程，默认添加 随机翻转
        if isTrain:
            transforms_compose.append(transforms.RandomVerticalFlip(0.5))
            transforms_compose.append(transforms.RandomHorizontalFlip(0.5))
        # 遍历每种变化的推理方式
        for _transforms_compose in _transform:
            if "CenterCrop" in _transforms_compose:
                _image_size = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.CenterCrop(_image_size))
            elif "Resize" in _transforms_compose:
                _image_size = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.Resize((_image_size, _image_size)))
            elif "RandomAffine" in _transforms_compose:
                transforms_compose.append(transforms.RandomAffine(0, (0.08, 0.08)))
            elif "ColorJitter" in _transforms_compose:
                transforms_compose.append(transforms.ColorJitter(0.2, 0.1, 0.1, 0.01))
            elif "RandomCrop" in _transforms_compose:
                _image_size = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.RandomCrop((_image_size, _image_size)))
            elif "Pad" in _transforms_compose:
                _pad_margin = int(_transforms_compose.split('-')[-1])
                transforms_compose.append(transforms.Pad(_pad_margin, padding_mode='reflect'))
        transforms_compose.append(Normalize())
        re_transforms.append(transforms.Compose(transforms_compose))
    return re_transforms

def train(model):
    train_sucess_flag = False

    # load classes
    binary_class = TrainVars.get('binary_class')
    multi_class = TrainVars.get('multi_class')

    # 数据集
    train_devices, dataset = split_devices_train_datas_ppl(TrainVars.get('dataset_path'), binary_class)
    if len(dataset[0]) == 0:
        train_log_write("dataset is null!")
        return train_sucess_flag, None
    train_images, train_labels, val_images, val_labels = dataset

    # train transform
    train_transforms_config = TrainVars.get('model_train_transform')
    train_transforms = transforms_set(train_transforms_config, isTrain=True)
    val_transforms_config = TrainVars.get('model_val_transform')
    val_transforms = transforms_set(val_transforms_config)

    # dataloader
    train_datas = TrainDataset(train_images, train_labels, train_transforms, multi_classs=multi_class)
    train_dataloader = torch.utils.data.DataLoader(train_datas, batch_size=TrainVars.get('batch_size'),
                                                   num_workers=TrainVars.get('cuda_num_workers'),
                                                   drop_last=True, shuffle=True, collate_fn=custom_collate_train)
    valid_datas = TrainDataset(val_images, val_labels, val_transforms, multi_classs=multi_class)
    valid_dataloader = torch.utils.data.DataLoader(valid_datas, batch_size=TrainVars.get('batch_size'),
                                                   num_workers=TrainVars.get('cuda_num_workers'),
                                                   drop_last=False, shuffle=False, collate_fn=custom_collate_train)

    train_log_write('----> 模型训练数据集准备完成！train data set: ' + str(len(train_datas)) + "; val data set: " + str(
        len(valid_datas)))
    loss_fn = LSR(use_focal_loss=TrainVars.get('use_focalloss')).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=TrainVars.get('lr'))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=TrainVars.get('factor'),
                                                              patience=TrainVars.get('patience'), verbose=True)
    _acc_los_path = os.path.join(TrainVars.get('logs_train_path') + "Temp", "acc_loss.jpg")
    _confusionmatrix_path = os.path.join(TrainVars.get('logs_train_path') + "Temp", "confusionmatrix.jpg")
    train_acc_loss = dict()
    val_acc_loss = dict()
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=binary_class)  # 实例化
    for epoch in range(TrainVars.get('epochs')):
        train_acc_avg, train_loss = train_epoch(train_dataloader, model, loss_fn, optimizer, epoch)
        val_acc_avg, val_loss = validate(valid_dataloader, model, loss_fn, drawconfusionmatrix)
        train_acc_loss[epoch] = [train_acc_avg, train_loss]
        val_acc_loss[epoch] = [val_acc_avg, val_loss]
        lr_scheduler.step(val_acc_avg)
        # 只有当精度超过0.95时才保存权重
        if val_acc_avg > TrainVars.get('acc_save_thresh') and epoch > 2:
            _save_path = TrainVars.get('model_path')
            torch.save(model.module.state_dict(), _save_path)
            train_log_write('--> save model to: ' + _save_path)
            train_sucess_flag = True
        # 绘制PR曲线
        draw_pr_curve(train_acc_loss, val_acc_loss, _acc_los_path)
    drawconfusionmatrix.drawMatrix(_confusionmatrix_path)
    return train_sucess_flag, train_devices

def train_epoch(train_dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_multi = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, target, multi_target) in enumerate(train_dataloader):
        if images is None:
            continue
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()
        multi_target = multi_target.cuda()

        optimizer.zero_grad()
        # compute y_pred
        y_pred, y_pred_multi = model(images)
        loss = criterion(y_pred, target)

        # multi_calss
        multi_target_index = torch.nonzero(multi_target == 99).squeeze().cpu().tolist()
        multi_target_new = del_tensor_ele(multi_target, multi_target_index)
        y_pred_multi_new = del_tensor_ele(y_pred_multi, multi_target_index)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(y_pred.data, target, topk=(1, 1))
        if multi_target_new.shape[0] != 0:
            prec1_multi, prec2_multi = accuracy(y_pred_multi_new.data, multi_target_new, topk=(1, 1))
            acc_multi.update(prec1_multi.item(), y_pred_multi_new.shape[0])

        losses.update(loss.item(), images.size(0))
        acc.update(prec1.item(), images.size(0))
        loss += criterion(y_pred_multi_new, multi_target_new)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2000 == 0:
            imshow_str = "Epoech: " + str(epoch) + "[" + str(i) + "]/[" + str(len(train_dataloader)) + "]: "
            imshow_str += "Time: " + str(round(batch_time.val, 3)) + "(" + str(round(batch_time.avg, 3)) + ") "
            imshow_str += "Data: " + str(round(data_time.val, 3)) + "(" + str(round(data_time.avg, 3)) + ") "
            imshow_str += "Loss: " + str(round(losses.val, 3)) + "(" + str(round(losses.avg, 3)) + ") "
            imshow_str += "Accuracy: " + str(round(acc.val, 3)) + "(" + str(round(acc.avg, 3)) + ")"
            imshow_str += "MultiAccuracy: " + str(round(acc_multi.val, 3)) + "(" + str(
                round(acc_multi.avg, 3)) + ")"
            train_log_write(imshow_str)
    return round(acc.avg / 100.0, 4), round(losses.val, 3)

def validate(val_loader, model, criterion, drawconfusionmatrix):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_multi = AverageMeter()
    model.eval()
    end = time.time()
    for i, (images, labels, multi_target) in enumerate(val_loader):
        if images is None:
            continue
        images = images.cuda()
        labels = labels.cuda()
        multi_target = multi_target.cuda()

        # compute y_pred
        with torch.no_grad():
            y_pred, y_pred_multi = model(images)
            loss = criterion(y_pred, labels)

            # multi_calss
            multi_target_index = torch.nonzero(multi_target == 99).squeeze().cpu().tolist()
            multi_target_new = del_tensor_ele(multi_target, multi_target_index)
            y_pred_multi_new = del_tensor_ele(y_pred_multi, multi_target_index)


        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))

        if multi_target_new.shape[0] != 0:
            prec1_multi, prec2_multi = accuracy(y_pred_multi_new.data, multi_target_new, topk=(1, 1))
            acc_multi.update(prec1_multi.item(), y_pred_multi_new.shape[0])
        losses.update(loss.item(), images.size(0))
        acc.update(prec1.item(), images.size(0))
        loss += 0.5 * criterion(y_pred_multi_new, multi_target_new)

        predict_np = np.argmax(y_pred.cpu().detach().numpy(), axis=-1)
        labels_np = labels.cpu().numpy()
        drawconfusionmatrix.update(predict_np, labels_np)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            imshow_str = "TrainVal: " + "[" + str(i) + "]/[" + str(len(val_loader)) + "]. "
            imshow_str += "Time: " + str(round(batch_time.val, 3)) + "(" + str(round(batch_time.avg, 3)) + ")"
            imshow_str += "Loss: " + str(round(losses.val, 3)) + "(" + str(round(losses.avg, 3)) + ")"
            imshow_str += "Accuracy: " + str(round(acc.val, 3)) + "(" + str(round(acc.avg, 3)) + ")"
            imshow_str += "MultiAccuracy: " + str(round(acc_multi.val, 3)) + "(" + str(
                round(acc_multi.avg, 3)) + ")"
            train_log_write(imshow_str)
    train_log_write(str("* Accuracy: " + str(round(acc.avg, 3))))
    return round(acc.avg / 100.0, 4), round(losses.val, 3)






