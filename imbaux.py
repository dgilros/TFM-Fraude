import tensorflow as tf

class CallableFactory:
    def __init__(self):
        self._builders = {}
        
    def register(self, key, builder, **kwargs):
        self._builders[key] = builder, kwargs
        
    def create(self, key, **kwargs):
        pair = self._builders.get(key)
        if not pair: raise ValueError(key)
        builder, args = pair
        args = dict(args, **kwargs)
        return builder(**args)
    
    def keys_list(self):
        return list(self._builders.keys())

class MLPClassifier:
    def __init__(self, **sk_args):
        pass            
            