import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class CNN(nn.Module):

    def __init__(self, features, img_size, num_classes, init_weights=True):

        super(CNN, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.epsilon = 1e-4

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        self.add_on_layers = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(in_features=25088, out_features=4096, bias=False),
            nn.Linear(in_features=100352, out_features=4096, bias=False),
            #nn.Linear(in_features=108192, out_features=4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2, bias=False)
            )
    
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def forward(self, x):
        logits = self.conv_features(x)
        return logits

    def __repr__(self):
        # CNN(self, features, img_size, num_classes, init_weights=True):
        rep = (
            'CNN(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.num_classes,
                          self.epsilon)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_CNN(base_architecture, pretrained=True, 
                   img_size=224, num_classes=2):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    return CNN(features=features,
                 img_size=img_size,
                 num_classes=num_classes,
                 init_weights=True)

