import os
from collections import defaultdict
from util.tools import read_yaml, get_project_path


client_list = ["jssi-Bumpping", "M6", "M24", "M47", "forehope"]


class bat_config:
    def __init__(self):
        self.yaml_path = os.path.join(get_project_path(), "config", "base")
        self.reload_yaml()
        self.reload_bat_path()

    def reload_yaml(self):
        deploy_cfg = read_yaml(os.path.join(self.yaml_path, "deploy.yaml"))
        train_cfg = read_yaml(os.path.join(self.yaml_path, "train.yaml"))

        self.client = deploy_cfg.get("client") if deploy_cfg.get("client") else train_cfg.get("client")
        assert self.client in client_list

        # 训练参数
        self.train_args = defaultdict(str)
        for key, value in train_cfg.items():
            if isinstance(value, int):
                value = str(value)
            self.train_args[key] = value

        self.reload_bat_path()

        return deploy_cfg, train_cfg

    def reload_bat_path(self):
        self.server_monitor_url = self.get_server_monitor_url()
        self.server_bat_path = self.get_server_bat()
        self.train_bat_path = self.get_train_bat()

    def get_server_monitor_url(self):
        url = "http://127.0.0.1:9527/monitor"
        if self.client == "jssi-Bumpping":
            url = "http://127.0.0.1:9527/monitor"
        elif self.client in ["M6", "M24", "M47"]:
            url = "http://127.0.0.1:9527/monitor"
        elif self.client == "forehope":
            url = "http://127.0.0.1:9527/monitor"
        return url

    def get_server_bat(self):
        bat_path = ''
        if self.client == "jssi-Bumpping":
            bat_path = r"D:\Solution\code\automatic_defect_classification_server\service\Run\jssi_Bumpping\ADC服务启动.bat"
        elif self.client in ["M6", "M24", "M47"]:
            bat_path = r"D:\Solution\code\smic\DCL\run\ADC服务启动.bat"
        elif self.client == "forehope":
            bat_path = ""
        return bat_path

    def get_train_bat(self):
        bat_path = ''
        if self.client == "jssi-Bumpping":
            bat_path = r"D:\Solution\code\DCL\run\模型迭代.bat"
        elif self.client in ["M6", "M24", "M47"]:
            bat_path = r"D:\Solution\code\smic\DCL\run\模型迭代.bat"
        elif self.client == "forehope":
            bat_path = r""
        return bat_path


class dataset_config():
    def __init__(self, client, val_ratio=0.1):
        self.client = client
        self.copy_threads = 20  # 复制图片线程数
        assert client in client_list
        self.dataset = {
            # M6
            "Back_M6":      {"train": "Back_M6",         "val": "Back_M6_val",
                             "categories": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3", 'BSCS': "4", "BSCSS": "5", "BSPS": "6"}},
            "BackDark_M6":  {"train": "BackDark_M6", "val": "BackDark_M6_val",
                             "categories": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3", 'BSCS': "4", "BSCSS": "5", "BSPS": "6"}},
            "Front_M6":     {"train": "Front_M6",        "val": "Front_M6_val",
                             "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3", 'PASD': "4", 'SINR': "5", "PASP": "6", "PANS": "7"}},
            "FrontDark_M6": {"train": "FrontDark_M6", "val": "FrontDark_M6_val",
                             "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3", 'PASD': "4", 'SINR': "5", "PASP": "6", "PANS": "7"}},
            # M24
            "Back_M24":     {"train": "Back_M24",        "val": "Back_M24_val",
                             "categories": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3"}},
            "BackDark_M24": {"train": "BackDark_M24",    "val": "BackDark_M24_val",
                             "categories": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3"}},
            "Front_M24":    {"train": "Front_M24",          "val": "Front_M24_val",
                             "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3"}},
            "FrontDark_M24": {"train": "FrontDark_M24",  "val": "FrontDark_M24_val",
                              "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3"}},
            # jssi-Bumpping
            "jssi-Bumpping_photo": {"train": "jssi-Bumpping_photo", "val": "jssi-Bumpping_photo_val",
                                    "categories": {"Bad": "0", "Good": "1"}},
            "jssi-Bumpping_aoi": {"train": "jssi-Bumpping_aoi", "val": "jssi-Bumpping_aoi_val",
                                    "categories": {"Bad": "0", "Good": "1"}},
        }

        self.val_ratio = val_ratio
