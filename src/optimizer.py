import numpy as np

class SGD:
    def __init__(self, lr=1e-4):
        self.lr=lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key]=params[key]-self.lr*grads[key]
            
class MomentumSGD:
    def __init__(self, lr=1e-4, moment=0.9):
        self.lr=lr
        self.moment=moment
        self.w_vec={}
    def update(self, params, grads):
        for k,v in params.items():
                self.w_vec.setdefault(k, np.zeros_like(v))
        
        for key in params.keys():
            params[key]=params[key]-self.lr*grads[key]+self.moment*self.w_vec[key]
            self.w_vec[key]=self.moment*self.w_vec[key]-self.lr*grads[key]
            
class Adam:
    def __init__(self, lr=1e-4, b1=0.9, b2=0.999, h=1e-8):
        self.lr=lr
        self.b1=b1
        self.b2=b2
        self.h=h
        self.step=0
        self.m={}
        self.v={}
        
    def update(self, params, grads):
        for k,v in params.items():
            self.m.setdefault(k, np.zeros_like(v))
            self.v.setdefault(k, np.zeros_like(v))
        
        self.step+=1
        tmp_lr=self.lr*np.sqrt(1.0-self.b2**self.step) / (1.0-self.b1**self.step)
        
        for key in params.keys():
            self.m[key]=self.m[key]+(1-self.b1) * (grads[key]-self.m[key])
            self.v[key]=self.v[key]+(1-self.b2) * (grads[key]**2-self.v[key])
            params[key]=params[key]-(tmp_lr*self.m[key] / (np.sqrt(self.v[key])+self.h))      