import logging.handlers
import logging


def logging_init(log_file):
    loger = logging.getLogger()
    # 设置日志回滚，每天(24小时)回滚一次，最大备份15个文件，若是生成新的会将最早的一个文件删除
    trf = logging.handlers.TimedRotatingFileHandler(log_file, when='H', interval=24, backupCount=15)
    trf.suffix = "%Y-%m-%d_%H-%M-%S.log"
    #trf.suffix = "%Y-%m-%d.log" #备份日志文件格式
    # 设置最低打印日志等级
    trf.setLevel(logging.DEBUG)
    # 设置日志格式
    format = logging.Formatter('%(asctime)s - %(message)s')
    trf.setFormatter(format)
    loger.addHandler(trf)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename=log_file)
