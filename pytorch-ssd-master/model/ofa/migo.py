import torch.nn as nn
import os
class MIGO(nn.Module):
    def __init__(self, mobilenet, restore=None):
        super(MIGO, self).__init__()
        self.backbone = mobilenet
        self.restore = restore
        if self.restore:self.resume()
        self.feature=[]
        self.feature.append(self.backbone.first_conv)
        for i, layer in enumerate(self.backbone.blocks):
            self.feature.append(layer)
        self.feature.append(self.backbone.final_expand_layer)
    def resume(self):
        if not os.path.isfile(self.restore):
            print('train migo-ssd from scratch')
        else:
            print('restore pretrained weights on ImageNet')

        state_dict = self.backbone.state_dict()
        pretrained = torch.load(self.restore)['state_dict_ema']
        new_state_dict = {}

        for k, v in state_dict.items():
            new_state_dict[k] = pretrained[k]
        state_dict.update(new_state_dict)
        print('restore pretrained done')

    def forward(selfs, x):
        return x
