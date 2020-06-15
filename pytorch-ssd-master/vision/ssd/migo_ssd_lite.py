import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import InvertedResidual
from ..nn.mobilenet_v3_ssd import MobileNetV3

from .ssd import SSD, GraphPath
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config
from model.ofa.migo import MIGO


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def create_migo_ssd_lite(num_classes, net_config, pretrain=None, size=0, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False ):
    # base_net=MobileNetV3(model_mode="LARGE").features
    from model.my_searched_model import my_specialized
    migo=my_specialized(num_classes=num_classes, net_config=net_config)
    base_net=nn.ModuleList(MIGO(mobilenet=migo, restore=pretrain).feature)
    # base_net.append(migo.first_conv)
    # for i, layer in enumerate(migo.blocks):
    #     base_net.append(layer)
    # base_net.append(migo.final_expand_layer)
    # print(base_net)
    in_channels=[[336,960], [432,1248], [432,1248]]
    # in_channels=[[240,960], [312,1248], [312,1248]]
    source_layer_indexes = [
        GraphPath(18, 'mobile_inverted_conv', -1),
        23,
    ]
    extras = ModuleList([
        InvertedResidual(in_channels[size][1], 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(in_channels[size][0] * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=in_channels[size][1], out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(in_channels[size][0] * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=in_channels[size][1], out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)



def create_migo_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
