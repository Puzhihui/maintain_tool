import os

from util.tools import read_yaml, get_project_path
from config.script_cfg import bat_config, dataset_config

# 加载yaml文件配置
deploy_cfg = read_yaml(os.path.join(get_project_path(), "config", "base", "deploy.yaml"))
train_cfg = read_yaml(os.path.join(get_project_path(), "config", "base", "train.yaml"))

# 加载server和train的bat config
client = deploy_cfg.get("client") if deploy_cfg.get("client") else train_cfg.get("client")
bat_cfg = bat_config(client)

# 数据集配置
dataset_cfg = dataset_config(client)
