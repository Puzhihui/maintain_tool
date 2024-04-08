# encoding:utf-8
from flask import Blueprint, request
import json
from util.tools import read_gpu_info
gpuinfo_api = Blueprint('gpuinfo_api', __name__)

# gpuinfo
@gpuinfo_api.route("/GPUInfo", methods=['POST'])
def GPU_info():
    if request.method == 'POST':
        request_param = request.get_json()
        results = {"ErrorCode": 1, "Msg": "option error!", "Data": None}
        if request_param['Option'] == "get":
            data = read_gpu_info()
            results = {"ErrorCode": 0, "Msg": "Success", "Data": data}
        return json.dumps(results)