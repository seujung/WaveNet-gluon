import os
from scipy.io import wavfile
from utils import *
from mxnet import gluon, autograd, nd

def load_wav(file_nm):
    fs, data = wavfile.read(os.getcwd()+'/data/'+file_nm)
    return  fs, data

def data_generation(data,framerate, seq_size, mu, ctx):
    #t = np.linspace(0,5,framerate*5)
    #data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    div = max(data.max(),abs(data.min()))
    data = data/div
    while True:
        start = np.random.randint(0,data.shape[0]-seq_size)
        ys = data[start:start+seq_size]
        ys = encode_mu_law(ys,mu)
        yield nd.array(ys[:seq_size],ctx=ctx)
        
def data_generation_sample(data, framerate, seq_size, mu, ctx):
    #t = np.linspace(0,5,framerate*5)
    #data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    div = max(data.max(),abs(data.min()))
    data = data/div
    start = 0
    ys = data[start:start+seq_size]
    ys = encode_mu_law(ys,mu)
    return nd.array(ys[:seq_size],ctx=ctx)