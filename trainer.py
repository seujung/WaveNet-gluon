import os, sys
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,utils 
import mxnet.ndarray as F
from tqdm import trange

from models import *
from utils import *
from data_loader import load_wav, data_generation, data_generation_sample

# set gpu count
def setting_ctx(use_gpu):
    if (use_gpu):
        ctx = mx.gpu()
    else :
        ctx = mx.cpu()
    return ctx

class Train(object):
    def __init__(self, config):
        ##setting hyper-parameters
        self.batch_size = config.batch_size
        self.epoches = config.epoches
        self.mu =  config.mu
        self.n_residue = config.n_residue
        self.n_skip = config.n_skip
        self.dilation_depth = config.dilation_depth
        self.n_repeat = config.n_repeat
        self.seq_size = config.seq_size
        self.use_gpu = config.use_gpu
        self.ctx = setting_ctx(self.use_gpu)
        self.build_model()
        
    def build_model(self):
        self.net = WaveNet(mu=self.mu, n_residue=self.n_residue, n_skip=self.n_skip, dilation_depth=self.dilation_depth, n_repeat=self.n_repeat)
        #parameter initialization
        self.net.collect_params().initialize(ctx=self.ctx)
        #set optimizer
        self.trainer = gluon.Trainer(self.net.collect_params(),optimizer='adam',optimizer_params={'learning_rate':0.01 })
        self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        
    def save_model(self,epoch,current_loss):
        filename = 'models/best_perf_epoch_'+str(epoch)+"_loss_"+str(current_loss)
        self.net.save_params(filename)
    
    def train(self):
        fs, data = load_wav('parametric-2.wav')
        g = data_generation(data,fs,mu=self.mu, seq_size=self.seq_size,ctx=self.ctx)
        
        loss_save = []
        best_loss = sys.maxsize
        for epoch in trange(self.epoches):
            loss = 0.0
            for _ in range(self.batch_size):
                batch = next(g)
                x = batch[:-1]
                with autograd.record():
                    logits = self.net(x)
                    sz = logits.shape[0]
                    loss = loss + self.loss_fn(logits, batch[-sz:])
                loss.backward()
                self.trainer.step(1,ignore_stale_grad=True)
            loss_save.append(nd.sum(loss).asscalar()/self.batch_size)
        
            #save the best model
            current_loss = nd.sum(loss).asscalar()/self.batch_size
            if best_loss > current_loss:
                print('epoch {}, loss {}'.format(epoch, nd.sum(loss).asscalar()/self.batch_size))
                self.save_model(epoch,current_loss)
                best_loss = current_loss
            
    def generate_slow(self, x, models, dilation_depth, n_repeat, ctx, n=100):
        dilations = [2**i for i in range(dilation_depth)] * n_repeat 
        res = list(x.asnumpy())
        for _ in trange(n):
            x = nd.array(res[-sum(dilations)-1:],ctx=ctx)
            y = models(x)
            res.append(y.argmax(1).asnumpy()[-1])
        return res
    
    def generation(self):
        fs, data = load_wav('parametric-2.wav')
        initial_data = data_generation_sample(data,fs,mu=self.mu, seq_size=3000,ctx=self.ctx)
        gen_rst = self.generate_slow(initial_data[0:3000],self.net,dilation_depth=10,n_repeat=2,n=2000,ctx=self.ctx)
        gen_wav = np.array(gen_rst)
        gen_wav = decode_mu_law(gen_wav, 128)
        np.save("wav.npy",gen_wav)
        
        
