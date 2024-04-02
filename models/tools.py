# encoding:utf-8
import hashlib
import os
import glob
import torch
import numpy as np
from PIL import Image, ImageFile
from matplotlib.ticker import MultipleLocator

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import torch.nn as nn
import torch.nn.functional as F
from shutil import copy
from models.vars import *
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# ---------------------1.归一化-----------------------#
class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = np.array(mean)[..., np.newaxis, np.newaxis]
        self.std = np.array(std)[..., np.newaxis, np.newaxis]
    def __call__(self, image):
        image = np.array(image)/255.0
        image = np.transpose(image, axes=(2, 0, 1))
        normalize_image = (image - self.mean) / self.std
        normalize_image = normalize_image.astype(np.float32)
        normalize_image = torch.from_numpy(normalize_image)
        return normalize_image

# 带focal loss 和 label smooth的交叉熵损失函数
class LSR(nn.Module):
    def __init__(self, e=0.1, gamma=2, reduction='mean', use_focal_loss=False):
        super().__init__()
        self.e = e
        self.gamma = gamma
        self.reduction = reduction
        self.use_focal_loss = use_focal_loss

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError(f'Expected input batchsize ({x.size(0)}) to match target batch_size({target.size(0)})')
        if x.dim() < 2:
            raise ValueError(f'Expected input tensor to have least 2 dimensions(got {x.size(0)})')
        if x.dim() != 2:
            raise ValueError(f'Only 2 dimension tensor are implemented, (got {x.size()})')

        smoothed_target = F.one_hot(target, x.size(1)) * (1 - self.e) + self.e / x.size(1)
        pt = F.softmax(x, dim=-1).clamp(1e-6)
        loss = -smoothed_target * torch.log(pt)  # label smooth cross entropy

        if self.use_focal_loss:
            loss *= (1.0 - pt) ** self.gamma  # focal loss factor

        loss = torch.sum(loss, dim=-1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_device_is_normal(device_info):
    results = {"errorcode": str(0), "msg": "success"}
    is_normal = True

    # check device id
    Illegal_keywords = device_info['Illegal_keywords']
    device = device_info['device']
    if len(Illegal_keywords) != 0:
        for _Illegal_keyword in Illegal_keywords:
            if _Illegal_keyword in device.lower():
                results['errorcode'] = -3
                results['msg'] = 'error device id!'
                results['Data'] = []
                is_normal = False

    # 检查图像数量是否超出上限
    if len(device_info['image_list']) > device_info['limit_images']:
        results['errorcode'] = -4
        results['msg'] = 'images num to many!'
        results['Data'] = []
        is_normal = False

    # check dir is exists
    # if not os.path.exists(device_dir):
    #     imshow_str = "!" * 10 + " image path access failed:" + device_id + "!" * 10
    #     queue_context.put(imshow_str)
    #     results['errorcode'] = -5
    #     results['msg'] = 'image path access failed!'
    #     results['Data'] = []
    #     is_normal = False
    return is_normal, results

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8-sig') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def split_devices_train_datas_ppl(device_path, binary_class):
    train_devices = [];train_images=[];train_labels=[];val_images=[];val_labels=[]
    temp_path = os.path.join(device_path, "temp")
    dataset_is_device = TrainVars.get('is_device')
    defect_classes = TrainVars.get('defect_class')
    image_format = TrainVars.get('image_format')
    suppliers = TrainVars.get('suppliers')

    image_suffixs = ["/*.png", "/*.jpg", "/*.jpeg"]

    _device_train_max_num = TrainVars.get('device_max_thresh')
    _device_train_min_num = TrainVars.get('device_min_thresh')
    train_rate = TrainVars.get('train_val_rate')

    for _supplier_name in suppliers:
        _supplier_path = os.path.join(device_path, _supplier_name)
        if not os.path.exists(_supplier_path):
            continue
        _device_or_class = os.listdir(_supplier_path)

        # device format
        if dataset_is_device == 1:
            retrain_devices = _device_or_class
            for device in retrain_devices:
                device_dir = os.path.join(_supplier_path, device)
                if not os.path.exists(device_dir): continue
                class1_path = os.path.join(device_dir, binary_class[0])
                class2_path = os.path.join(device_dir, binary_class[1])
                if not os.path.exists(class1_path) or not os.path.exists(class2_path):continue
                device_class1_image_num = len(glob.glob(class1_path + '/*.jpg')) + len(glob.glob(class1_path + '/*.jpeg')) + len(glob.glob(class1_path + '/*.png'))
                device_class2_image_num = len(glob.glob(class2_path + '/*.jpg')) + len(glob.glob(class2_path + '/*.jpeg')) + len(glob.glob(class2_path + '/*.png'))
                if device_class1_image_num < _device_train_min_num or device_class2_image_num < _device_train_min_num:
                    continue
                train_images_num = min(device_class1_image_num, device_class2_image_num)
                train_images_num = min(train_images_num, _device_train_max_num)
                train_devices.append(device)
                for idx_label, _class in enumerate(binary_class):
                    defect_images = []
                    for _extension in image_suffixs:
                        defect_glob = os.path.join(device_dir, _class) + _extension
                        defect_images.extend(glob.glob(defect_glob))
                    random.shuffle(defect_images)
                    for index, per_image_address in enumerate(defect_images[:train_images_num]):
                        if index < int(train_images_num * train_rate):
                            train_images.append(per_image_address)
                            train_labels.append(idx_label)
                        else:
                            val_images.append(per_image_address)
                            val_labels.append(idx_label)
        # class format
        else:
            classes = _device_or_class
            for idx_label, _class in enumerate(classes):
                train_devices.append(_class)
                defect_images = []
                for _extension in image_suffixs:
                    defect_glob = os.path.join(_supplier_path, _class) + _extension
                    defect_images.extend(glob.glob(defect_glob))
                random.shuffle(defect_images)
                for index, per_image_address in enumerate(defect_images):
                    if index < int(len(defect_images) * train_rate):
                        train_images.append(per_image_address)
                        train_labels.append(idx_label)
                    else:
                        val_images.append(per_image_address)
                        val_labels.append(idx_label)

    # temp数据加入训练集
    for idx_label, _class in enumerate(binary_class):
        image_nums = len(os.listdir(os.path.join(temp_path, _class)))
        if image_nums == 0:
            continue
        defect_images = []
        for _extension in image_suffixs:
            defect_glob = os.path.join(temp_path, _class) + _extension
            defect_images.extend(glob.glob(defect_glob))
        random.shuffle(defect_images)
        for index, per_image_address in enumerate(defect_images):
            train_images.append(per_image_address)
            train_labels.append(idx_label)


    dataset = [train_images, train_labels, val_images, val_labels]
    return train_devices, dataset

def del_tensor_ele(arr, indexs):
    cut_num = 0
    if isinstance(indexs, int):
        indexs = [indexs]
    for _index in indexs:
        _index -= cut_num
        arr1 = arr[0:_index]
        arr2 = arr[_index+1:]
        arr = torch.cat((arr1, arr2), dim=0)
        cut_num += 1
    return arr

def accuracy(y_pred, y_actual, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def trans_device_image(image_paths):
    while True:
        image_path = image_paths.get()
        if image_path == None:break
        try:
            copy(image_path[0], image_path[1])
            os.remove(image_path[0])
        except:
            pass

def delete_image(images):
    while True:
        _image = images.get()
        if _image is None:break
        try:
            os.remove(_image)
        except:
            pass

def transfer_images(images):
    while True:
        _info = images.get()
        if _info is None:break

        source_path = _info[0]
        target_path = _info[1]
        try:
            copy(source_path, target_path)
        except:
            pass

def get_md5_str(file):
    md5_obj = hashlib.md5()
    with open(file, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            md5_obj.update(data)
    md5_str = str(md5_obj.hexdigest())
    return md5_str

def interval_day(start_date_str):
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d")
    today_datetime = datetime.now()
    return (today_datetime - start_datetime).days

def waferinfo_analysis(folder_text):
    result = "unkonw"
    try:
        last_info = folder_text.rsplit("/")[::-1]
        for _info in last_info:
            if "-" in _info:
                result = _info
                break
    except:
        pass
    return result


def calculate_area_threshold(area_value):
    digit = 0
    area = int(area_value)
    while area > 0:
        area //= 10
        digit += 1
    area = int(area_value)
    area_min = int(10**(digit-1))
    area_max = int(10**digit)
    area_threshold = area_min if area_max//2 > area else area_max//2
    return area_threshold

def draw_pr_curve(train_acc_loss, val_acc_loss, save_path):
    plt.cla()
    ax = plt.gca()  # 获取边框
    ax.xaxis.set_major_locator(MultipleLocator(1))
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    for i in range(len(train_acc_loss)):
        train_acc, train_loss = train_acc_loss[i]
        val_acc, val_loss = val_acc_loss[i]
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        eval_losses.append(val_loss)
        eval_acces.append(val_acc)
    # 3.acc & loss curve
    plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
    plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
    plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.plot(np.arange(len(eval_acces)), eval_acces, label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.title(' Acc & Loss curve')
    plt.savefig(save_path)  # 图片保存


class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[label, predict] += 1

    def getMatrix(self, normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换
            self.matrix = np.around(self.matrix, 2)  # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self, save_name=""):
        plt.cla()
        matrix_number = self.matrix.copy()
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("ADC-ZKFC Normalized confusion matrix", fontsize=16)  # title
        plt.xlabel("Predict label", fontsize=16)
        plt.ylabel("Truth label", fontsize=16)
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y - 0.12, value, verticalalignment='center', horizontalalignment='center',
                         fontsize=15)  # 写值
                plt.text(x, y + 0.12, int(matrix_number[y, x]), va='center', ha='center', fontsize=15)  # 显示对应的数字

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig(save_name, bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        # plt.show()

    def calculate_PR(self):
        matrix_number = self.matrix.copy()
        a = float(matrix_number[0, 0] + matrix_number[0, 1])
        b = float(matrix_number[0, 0])

        P = round(float(matrix_number[0, 0]) / float(matrix_number[0, 0] + matrix_number[0, 1]), 3)
        try:
            R = round(float(matrix_number[0, 0]) / float(matrix_number[0, 0] + matrix_number[1, 0]), 3)
        except:
            R = 0.0
        return P, R

def combine_image(origin_image, defect_image):
    w, h, _ = origin_image.shape
    defect_w, defect_h, _ = defect_image.shape
    start_x = w//2 - defect_w//2; start_y = h//2 - defect_h//2
    for _x in range(defect_w):
        for _y in range(defect_h):
            if sum(defect_image[_x, _y, :]) > 150:
                continue
            else:
                origin_image[_x + start_x, _y + start_y, :] = defect_image[_x, _y, :]
    return origin_image
