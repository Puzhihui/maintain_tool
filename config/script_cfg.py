client_list = ["jssi-Bumping", "M6", "M24", "M47", "forehope"]


class bat_config:
    def __init__(self, client):
        assert client in client_list
        self.client = client
        self.server_bat_path = self.get_server_bat()
        self.train_bat_path = self.get_train_bat()

    def get_server_bat(self):
        bat_path = ''
        if self.client == "jssi-Bumping":
            bat_path = ""
        elif self.client in ["M6", "M24", "M47"]:
            bat_path = r"D:\Solution\code\smic\DCL\run\ADC服务启动.bat"
        elif self.client == "forehope":
            bat_path = ""
        return bat_path

    def get_train_bat(self):
        bat_path = ''
        if self.client == "jssi-Bumping":
            bat_path = ""
        elif self.client in ["M6", "M24", "M47"]:
            bat_path = r"D:\Solution\code\smic\DCL\run\模型迭代.bat"
        elif self.client == "forehope":
            bat_path = r""
        return bat_path


class dataset_config():
    def __init__(self, client, val_ratio=0.1):
        self.client = client
        assert client in client_list
        self.supplier2dataset = {"FrontSide_Bright": {"train": "Front_{}".format(client), "val": "Front_{}_val".format(client),
                                                      "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "false": "3", 'PASD': "4", 'SINR': "5", "PASP": "6"}},
                                 "FrontSide_Dark":   {"train": "FrontDark_{}".format(client), "val": "FrontDark_{}_val".format(client),
                                                      "categories": {'PADC': "0", 'PAOH': "1", 'PASC': "2", "false": "3", 'PASD': "4", 'SINR': "5", "PASP": "6"}},
                                 "BackSide_Bright":  {"train": "Back_{}".format(client), "val": "Back_{}_val".format(client),
                                                      "categories": {'BSDC': "0", 'BSOH': "1", 'scratch': "2", "false": "3", 'BSCS': "4"}},
                                 "BackSide_Dark":    {"train": "BackDark_{}".format(client), "val": "BackDark_{}_val".format(client),
                                                      "categories": {'BSDC': "0", 'BSOH': "1", 'scratch': "2", "false": "3", 'BSCS': "4"}},
                                 "jssi_aoi":         {"train": "jssi-Bumping"}}

        self.val_ratio = val_ratio

