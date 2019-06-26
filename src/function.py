import numpy as np
import os


#relu関数
def relu(x):
    return np.maximum(0, x)
#sigmoid関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#binary cross entropy損失関数の定義
def binary_cross_entropy_loss(s, y):
    return y*np.log(1+np.exp(-s))+(1-y)*np.log(1+np.exp(s))

#数値微分の定義　h=1e-3
def numerical_gradient(f, x):
    h = 1e-3 # 0.001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def train_data_loader(path=os.getcwd(), file_name=False):
    chdir=os.getcwd()
    path=os.path.join(path, "datasets", "train")
    os.chdir(path)
    graphs=[file for file in os.listdir('.') if file.endswith('graph.txt')]
    labels=[file for file in os.listdir('.') if file.endswith('label.txt')]
    dataset=dict(zip(sorted(graphs),sorted(labels)))

    valid_num=list(np.random.choice(np.arange(2000),400,replace=False))

    train_set, valid_set={}, {}
    for k,v in dataset.items():
        if int(k.split("_")[0]) in valid_num:
            valid_set[k]=v
        else:
            train_set[k]=v
    train_data=make_data(train_set, file_name=file_name)
    valid_data=make_data(valid_set, file_name=file_name)
    os.chdir(chdir)
    
    return train_data, valid_data

def test_data_loader(path=os.getcwd()):
    chdir=os.getcwd()
    test_data=[]
    path=os.path.join(path, "datasets", "test")
    os.chdir(path)
    graphs=[file for file in os.listdir('.') if file.endswith('graph.txt')]
    for file in graphs:
        with open(file) as graph:
            graph_lines=graph.readlines()
            D=int(graph_lines[0].rstrip("\n"))
            arr=[]
            for graph_line in graph_lines[1:]:
                graph_line=graph_line.rstrip("\n")
                arr.append(list(map(int, graph_line.split())))
            arr=np.array(arr)
            tmp_set=(D, arr, file)
            test_data.append(tmp_set)
    os.chdir(chdir)
    return test_data
    
def make_data(data_set, file_name=False):
    data=[]
    for k,v in data_set.items():
        with open(k) as graph, open(v) as label:
            graph_lines=graph.readlines()
            label_line=label.read().rstrip("\n")
            D=int(graph_lines[0].rstrip("\n"))
            arr=[]
            for graph_line in graph_lines[1:]:
                graph_line=graph_line.rstrip("\n")
                arr.append(list(map(int,graph_line.split())))
            arr=np.array(arr)
            if file_name:
                tmp_set=(D, arr, int(label_line), (k,v))
            else:
                tmp_set=(D, arr, int(label_line))
            data.append(tmp_set)
    return data

def get_accuracy(data, model):
    """
    引数としてtest_data_loaderで作成したデータとmodel.pyのGNNモデルをとり、データにおけるaccuracyを返す。
    """
    cnt=0
    for d in data:
        D, adj, y=d
        label=model.get_label(adj)
        if y==label:
            cnt+=1
    acc=cnt/len(data)
    return acc

