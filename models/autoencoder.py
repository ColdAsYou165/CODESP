import torch
import torch.nn as nn
import os
import pathlib
os.makedirs("../mywar/saad/happy",exist_ok=True)
def  happy(a):
    a+=1
    print(a)
b=1
b.apply(happy)
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        