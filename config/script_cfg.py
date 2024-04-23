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
        assert client in client_list
        self.supplier2dataset = {
            "FrontSide_Bright": {"train": "Front_{}".format(client), "val": "Front_{}_val".format(client),
                                 "categories_M6":  {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3", 'PASD': "4", 'SINR': "5", "PASP": "6", "PANS": "7"},
                                 "categories_M24": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3"},
                                 },
            "FrontSide_Dark":   {"train": "FrontDark_{}".format(client), "val": "FrontDark_{}_val".format(client),
                                 "categories_M6":  {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3", 'PASD': "4", 'SINR': "5", "PASP": "6", "PANS": "7"},
                                 "categories_M24": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3"},
                                 },
            "BackSide_Bright":  {"train": "Back_{}".format(client), "val": "Back_{}_val".format(client),
                                 "categories_M6":  {'BSDC': "0", 'BSNS': "1", 'SCRATCH': "2", "FALSE": "3", 'BSCS': "4", "BSCSS": "5"},
                                 "categories_M24": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3"},
                                 },
            "BackSide_Dark":    {"train": "BackDark_{}".format(client), "val": "BackDark_{}_val".format(client),
                                 "categories_M6":  {'BSDC': "0", 'BSNS': "1", 'SCRATCH': "2", "FALSE": "3", 'BSCS': "4", "BSCSS": "5"},
                                 "categories_M24": {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3"},
                                 },

            "jssi_Bumpping_aoi": {"train": "jssi-Bumping"}}

        self.val_ratio = val_ratio
