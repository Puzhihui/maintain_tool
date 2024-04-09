from config.script_cfg import bat_config, dataset_config

# 加载server和train的bat config
bat_cfg = bat_config()

# 数据集配置
dataset_cfg = dataset_config(bat_cfg.client)
