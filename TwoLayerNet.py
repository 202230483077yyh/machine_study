import numpy as np

#激活函数，实现非线性变化
class Sigmoid:
    def __init__(self):
        self.params=[]

    def forward(slef,x):
        return 1/(1+np.exp(-x))
    
#全连接层，实现到下一层的神经元转化
class Affine:
    #初始化接收权重，偏置
    def __init__(self,W,b):
        self.params =[W,b]

    #具体转化下一层
    def forward(self ,x):
        W,b=self.params
        out =np.dot(x,W)+b
        return out
    
#封装两层神经网络
class TwoLayerNet:
    #获取各层神经元数目
    def __init__(self,input_size,hidden_size,out_put_size):
        I,H,O=input_size,hidden_size,out_put_size

        W1=np.random.randn(I,H)
        b1=np.random.randn(H)
        W2=np.random.randn(H,O)
        b2=np.random.randn(O)
        #将各层操作集中起来
        self.layers =[
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        #将各层的参数集中管理
        self.params =[]
        for layers in self.layers:
            self.params=layers.params

    #实现逐层前推
    def predict(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x

    
