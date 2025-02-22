import torch.nn as nn
from torch.autograd import Variable

from ptsemseg.models.fusion import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.utils import mutualinfo_entropy, plotEverything, plotPrediction


class CAFnet(nn.Module):
    def __init__(self,
                 backbone="segnet",
                 n_classes=21,
                 in_channels=3,
                 mcdo_passes=1,
                 dropoutP=0.1,
                 full_mcdo=False,
                 temperatureScaling=False,
                 freeze_seg=True,
                 freeze_temp=True,
                 fusion_module="1.3",
                 scaling_module="None",
                 pretrained_rgb="./models/Segnet/rgb_Segnet/rgb_segnet_mcdo_airsim_T000+T050.pkl",
                 pretrained_d="./models/Segnet/d_Segnet/d_segnet_mcdo_airsim_T000+T050.pkl"
                 ):
        super(CAFnet, self).__init__()

        self.rgb_segnet = segnet_mcdo(modality = 'rgb',
                                      n_classes=n_classes,
                                      mcdo_passes=mcdo_passes,
                                      dropoutP=dropoutP,
                                      full_mcdo=full_mcdo,
                                      in_channels=in_channels,
                                      temperatureScaling=temperatureScaling,
                                      freeze_seg=freeze_seg,
                                      freeze_temp=freeze_temp, )

        self.d_segnet = segnet_mcdo(modality = 'd',
                                    n_classes=n_classes,
                                    mcdo_passes=mcdo_passes,
                                    dropoutP=dropoutP,
                                    full_mcdo=full_mcdo,
                                    in_channels=in_channels,
                                    temperatureScaling=temperatureScaling,
                                    freeze_seg=freeze_seg,
                                    freeze_temp=freeze_temp, )

        self.rgb_segnet = torch.nn.DataParallel(self.rgb_segnet, device_ids=range(torch.cuda.device_count()))
        self.d_segnet = torch.nn.DataParallel(self.d_segnet, device_ids=range(torch.cuda.device_count()))

        # initialize segnet weights
        if pretrained_rgb:
            self.loadModel(self.rgb_segnet, pretrained_rgb)
        if pretrained_d:
            self.loadModel(self.d_segnet, pretrained_d)

        # freeze segnet networks
        for param in self.rgb_segnet.parameters():
            param.requires_grad = False
        for param in self.d_segnet.parameters():
            param.requires_grad = False

        self.fusion = self._get_fusion_module(fusion_module, n_classes)
        self.bn_rgb = nn.Sequential(nn.BatchNorm2d(1),
                                    nn.ReLU())
        self.bn_d = nn.Sequential(nn.BatchNorm2d(1),
                                  nn.ReLU())

        if hasattr(self.d_segnet.module, 'temperature'):
            self.scale_d = self._get_scale_module(scaling_module, bias_init=self.d_segnet.module.temperature)
            self.scale_rgb = self._get_scale_module(scaling_module, bias_init=self.rgb_segnet.module.temperature)
        else:
            self.scale_d = self._get_scale_module(scaling_module)
            self.scale_rgb = self._get_scale_module(scaling_module)
        self.i = 0

        self.normalize = False

    def forward(self, inputs):

        # Freeze batchnorm
        self.rgb_segnet.eval()
        self.d_segnet.eval()

        inputs_rgb = inputs[:, :3, :, :]
        inputs_d = inputs[:, 3:, :, :]

        mean = {}
        variance = {}
        entropy = {}
        mutual_info = {}

        # computer logits and uncertainty measures
        mean['rgb'], variance['rgb'], entropy['rgb'], mutual_info['rgb'] = self.rgb_segnet.module.forwardMCDO(inputs_rgb)
        mean['d'], variance['d'], entropy['d'], mutual_info['d'] = self.d_segnet.module.forwardMCDO(inputs_d)

        variance['rgb'] = torch.mean(variance['rgb'], 1).unsqueeze(1)
        variance['d'] = torch.mean(variance['d'], 1).unsqueeze(1)

        if self.scale_d is not None:
            mean['rgb'] = self.scale_rgb(mean['rgb'], variance['rgb'], entropy['rgb'], mutual_info['rgb'])
            mean['d'] = self.scale_d(mean['d'], variance['d'], entropy['d'], mutual_info['d'])  # [bs, n, 512, 512]

        # fuse outputs
        x = self.fusion(mean, variance, entropy, mutual_info)  # [bs, n, 512, 512]

        # plot uncertainty
        self.i += 1
        # if (self.i) % 5 == 0:
        # p = nn.Softmax(dim=1)(x)
        # entropy['rgbd'], mutual_info['rgbd'] = mutualinfo_entropy(p.unsqueeze(-1))

        # pred = {}
        # pred['rgb'] = nn.Softmax(dim=1)(mean['rgb']).max(1)[0]
        # pred['d'] = nn.Softmax(dim=1)(mean['d']).max(1)[0]
        # pred['rgbd'] = p.max(1)[0]

        # labels = ['mutual info', 'entropy', 'probability', 'variance']

        # values = [mutual_info['rgb'], entropy['rgb'], pred['rgb'], variance['rgb'].squeeze(1)]
        # plotEverything('./plots/', self.i, self.i, "/rgb", values, labels)

        # values = [mutual_info['d'], entropy['d'], pred['d'], variance['d'].squeeze(1)]
        # plotEverything('./plots/', self.i, self.i, "/d", values, labels)

        # values = [mutual_info['rgbd'], entropy['rgbd'], pred['rgbd'], torch.zeros((1, 512, 512))]
        # plotEverything('./plots/', self.i, self.i, "/rgbd", values, labels)

        # inputs = {'rgb': inputs_rgb, 'd': inputs_d}
        # plotPrediction('./plots/', None, 11, self.i, self.i, "/inputs", inputs, torch.zeros((1, 512, 512)),
        # torch.zeros((1, 512, 512)))

        return x

    def loadModel(self, model, path):
        model_pkl = path

        print(path)
        if os.path.isfile(model_pkl):
            pretrained_dict = torch.load(model_pkl)['model_state']
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
        else:
            print("model not found")
            exit()

    def _get_fusion_module(self, name, n_classes=11):

        name = str(name)

        return {
            "1.0": GatedFusion(n_classes),
            "1.1": ConditionalAttentionFusion(n_classes),
            "1.3": UncertaintyGatedFusion(n_classes),
            "1.4": FullyUncertaintyGatedFusion(n_classes),

            "GatedFusion": GatedFusion(n_classes),
            "CAF": ConditionalAttentionFusion(n_classes),
            "UncertaintyGatedFusion": UncertaintyGatedFusion(n_classes),

            "Average": Average(n_classes),
            "Multiply": Multiply(n_classes),
            "NoisyOr": NoisyOr(n_classes),
        }[name]

    def _get_scale_module(self, name, n_classes=11, bias_init=None):

        name = str(name)

        return {
            "temperature": TemperatureScaling(n_classes, bias_init),
            "uncertainty": UncertaintyScaling(n_classes, bias_init),
            "LocalUncertaintyScaling": LocalUncertaintyScaling(n_classes, bias_init),
            "GlobalUncertainty": GlobalUncertaintyScaling(n_classes, bias_init),
            "GlobalLocalUncertainty": GlobalLocalUncertaintyScaling(n_classes, bias_init),
            "None": None
        }[name]
