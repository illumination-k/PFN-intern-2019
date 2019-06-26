import numpy as np
import os
from function import *

class GNN:
    """
    引数としてDの次元を取り、aggregateの回数と学習器θの初期条件を設定する。
    初期の特徴ベクトルx_0は正規分布に従った乱数行列を設定する。
    """
    def __init__(self, D, T=2, init_mean=0, init_std=0.4, init_b=0):
        self.D=D
        self.T=T
        self.x_vecs=None

        self.theta={}
        self.theta['W']=np.random.normal(init_mean ,init_std,(D,D))
        self.theta['A']=np.random.normal(init_mean ,init_std,(1,D))
        self.theta['b']=np.array([init_b])
        
    def aggregate(self, adj, initilize=True):
        np.random.seed(201)
        if initilize:
            self.x_vecs=np.random.randn(self.D, len(adj))
        #集約-1
        a=np.zeros_like(self.x_vecs.T)
        for i in range(len(adj)):
            a[i]=self.x_vecs.T[adj[i]==1].sum(axis=0)
        #集約-2
        x=relu(self.theta['W']@a.T) #転置しておいたので計算処理上直す
        return x
    
    def get_h_G(self, adj, T=2, initilize=True):
        self.T=T
        self.x_vecs=self.aggregate(adj, initilize=initilize)
        if self.T>1:
            for i in range(self.T-1):
                self.x_vecs=self.aggregate(adj, initilize=False)
        #h_Gを得る
        h_G=np.sum(self.x_vecs, axis=1)
        return h_G
    
    def affine(self, adj):
        """
        隣接行列を引数に取り、h_Gの重み和を返す
        """
        #A@h_G+b
        h_G=self.get_h_G(adj)
        return self.theta['A']@h_G+self.theta['b']
    
    def get_label(self, adj):
        """
        隣接行列を引数にとり、modelの予測ラベルを返す
        """
        p=sigmoid(self.affine(adj))
        if p[0]>0.5:
            return 1
        else:
            return 0
    
    def loss(self, adj, y):
        """
        隣接行列と正解ラベルを引数に取り、損失関数の結果を返す
        """
        s=self.affine(adj)
        return binary_cross_entropy_loss(s, y)
    
    def numerical_gradient(self, adj, y):
        """
        隣接行列と正解ラベルを引数に取り、数値微分による勾配の辞書を返す
        """
        loss_theta=lambda theta: self.loss(adj, y)
        
        grads={}
        grads['W']=numerical_gradient(loss_theta, self.theta['W'])
        grads['A']=numerical_gradient(loss_theta, self.theta['A'])
        grads['b']=numerical_gradient(loss_theta, self.theta['b'])
        
        return grads

    def save(self, path=os.getcwd(), model_name="model"):
        """
        現時点でのモデルを保存する。pathに保存ディレクトリを、model_nameにモデルの名前をとる
        """
        chdir=os.getcwd()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        txt_name=model_name+".txt"
        with open(txt_name, "w") as f:
            line=str(self.D)
            f.write(line)
        npz_name=model_name+".npz"
        np.savez_compressed(npz_name, W=self.theta["W"], A=self.theta["A"], b=self.theta["b"])
        os.chdir(chdir)
    
    def load(self, path=os.getcwd(), model_name="model"):
        """
        保存されたモデルのパラメーターをロードする。pathに保存ディレクトリを、model_nameにモデルの名前をとる
        """
        chdir=os.getcwd()
        os.chdir(path)
        txt_name=model_name+".txt"
        npz_name=model_name+".npz"
        with open(txt_name) as f:
            lines=f.read()
            self.D=int(lines)
        loaded_arr=np.load(npz_name)
        for key in self.theta.keys():
            self.theta[key]=loaded_arr[key]
        os.chdir(chdir)