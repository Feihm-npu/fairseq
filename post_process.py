import sys
from numpy import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.cm as cm

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_name', type=str, default="hotness_gpu_0.npy")
        parser.add_argument('--num_steps', type=int, default=100)
        parser.add_argument('--num_tokens', type=int, default=2048)
        parser.add_argument('--num_local_experts', type=int, default=512)
        parser.add_argument('--num_moe_layers', type=int, default=6)
        parser.add_argument('--dataset', type=str, default="Pubmed")
        args = parser.parse_args()
        
        file_name = args.file_name
        num_steps = args.num_steps
        num_tokens = args.num_tokens
        num_local_experts = args.num_local_experts
        num_moe_layers = args.num_moe_layers 
        dataset = args.dataset

        #load array
        data = load(file_name)
        data = data.reshape((num_steps+1, num_moe_layers*num_tokens*2))
        data = data.reshape((num_steps+1, num_moe_layers, num_tokens*2))
        
        dict_ = {}
        out_ = {}
        for i in range(1, num_moe_layers+1):
            dict_['layer%s' % i] = data[:,i-1,:]
            out_['out%s' % i] = np.zeros([num_steps+1, num_local_experts])
            for j in range(0, num_steps+1):
                out_['out%s' % i][j] = np.bincount(dict_['layer%s' % i][j].astype(int), minlength=num_local_experts)
        
            interp = 'bilinear'
            fig, axs = plt.subplots(1,1)
            title_ = str(dataset + 'Layer_' + str(i)) 
            axs.imshow(out_['out%s' % i], origin='lower', cmap='Blues', interpolation=interp)
            axs.set_xlabel('Experts [0-512]')
            axs.set_ylabel('Input Batches [0-100]')
            axs.axis('scaled')
            fig.suptitle(title_, fontsize=16)
            fig.set_figheight(3)
            fig.set_figwidth(8)
            fig.tight_layout()
            fig.colorbar(cm.ScalarMappable(cmap='Blues'), ax=axs)
            plt.show()
            plt.savefig(dataset + '_' + 'Layer_' + str(i) + '.jpg')

