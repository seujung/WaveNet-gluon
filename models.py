import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,utils 
import mxnet.ndarray as F

class One_Hot(nn.Block):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        
    def forward(self, X_in):
        with X_in.context:
            X_in = X_in
            self.ones = nd.one_hot(nd.arange(self.depth),self.depth)
            return self.ones[X_in,:]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class WaveNet(nn.Block):
    def __init__(self, mu=256,n_residue=32, n_skip= 512, dilation_depth=10, n_repeat=5):
        # mu: audio quantization size
        # n_residue: residue channels
        # n_skip: skip channels
        # dilation_depth & n_repeat: dilation layer setup
        super(WaveNet, self).__init__()
        self.dilation_depth = dilation_depth
        self.dilations = [2**i for i in range(dilation_depth)] * n_repeat      
        with self.name_scope():
            self.one_hot = One_Hot(mu)
            self.from_input = nn.Conv1D(in_channels=mu, channels=n_residue, kernel_size=1)
            self.conv_sigmoid = nn.Sequential()
            self.conv_tanh = nn.Sequential()
            self.skip_scale = nn.Sequential()
            self.residue_scale = nn.Sequential()
            for d in self.dilations:
                self.conv_sigmoid.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=2, dilation=d))
                self.conv_tanh.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=2, dilation=d))
                self.skip_scale.add(nn.Conv1D(in_channels=n_residue, channels=n_skip, kernel_size=1, dilation=d))
                self.residue_scale.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=1, dilation=d))
            self.conv_post_1 = nn.Conv1D(in_channels=n_skip, channels=n_skip, kernel_size=1)
            self.conv_post_2 = nn.Conv1D(in_channels=n_skip, channels=mu, kernel_size=1)
        
    def forward(self,x):
        with x.context:
            output = self.preprocess(x)
            skip_connections = [] # save for generation purposes
            for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale):
                output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
                skip_connections.append(skip)
            # sum up skip connections
            output = sum([s[:,:,-output.shape[2]:] for s in skip_connections])
            output = self.postprocess(output)
        return output
        
    def preprocess(self, x):
        output = F.transpose(self.one_hot(x).expand_dims(0),axes=(0,2,1))
        output = self.from_input(output)
        return output

    def postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)
        output = self.conv_post_2(output)
        output = nd.reshape(output,(output.shape[1],output.shape[2]))
        output = F.transpose(output,axes=(1,0))
        return output
    
    def residue_forward(self, x, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = x
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + x[:,:,-output.shape[2]:]
        return output, skip
    
    def generate_slow(self, x, n=100):
        with x.context:
            res = list(x.asnumpy())
            for _ in range(n):
                x_ = nd.array(res[-sum(self.dilations)-1:])
                y = self.forward(x_)
                #_, i = y.max(dim=1)
                res.append(y.argmax(1).asnumpy()[-1])
        return res
