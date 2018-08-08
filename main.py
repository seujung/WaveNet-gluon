import mxnet as mx
import argparse
from trainer import Train

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--mu', type=int, default=128)
    parser.add_argument('--n_residue', type=int, default=24)
    parser.add_argument('--n_skip', type=int, default=128)
    parser.add_argument('--dilation_depth', type=int, default=10)
    parser.add_argument('--n_repeat', type=int, default=2)
    parser.add_argument('--seq_size', type=int, default=20000)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--generation', type=bool, default=True)
    config = parser.parse_args()
    
    trainer = Train(config)
    
    trainer.train()
    if (config.generation):
        trainer.generation()

if __name__ =="__main__":
    main()