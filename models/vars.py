# encoding:utf-8# --------------------- 存放server var class-----------------------#class ServerVars:    _data = {}    @classmethod    def set(cls, key, value):        cls._data[key] = value    @classmethod    def get(cls, key):        return cls._data.get(key)    @classmethod    def delete(cls, key):        if key in cls._data:            del cls._data[key]# ---------------------  train var class-----------------------#class TrainVars:    _data = {}    @classmethod    def set(cls, key, value):        cls._data[key] = value    @classmethod    def get(cls, key):        return cls._data.get(key)    @classmethod    def delete(cls, key):        if key in cls._data:            del cls._data[key]