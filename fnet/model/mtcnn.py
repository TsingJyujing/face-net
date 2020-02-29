"""
Author: Yuan Yifan <tsingjyujing@163.com>
In this file, we defined:
1. The Networks will be used in MTCNN
2. The loss function of output
3. Some utility function and layers (should move to utility later)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class MTCNNLoss(nn.Module):
    def __init__(self, w_detection: float, w_bbox: float, w_landmark: float):
        super().__init__()
        self.w_detection = w_detection
        self.w_bbox = w_bbox
        self.w_landmark = w_landmark

    def forward(
            self,
            y_pred: torch.FloatTensor,
            y_real: torch.FloatTensor,
            sample_weight: torch.FloatTensor
    ):
        """
        Get loss of the whole network
        :param sample_weight:
        :param y_pred:
        :param y_real:
        :return:
        """
        return mtcnn_loss(
            y_pred, y_real, sample_weight,
            self.w_detection,
            self.w_bbox,
            self.w_landmark
        )


def mtcnn_loss(
        y_pred: torch.FloatTensor,
        y_real: torch.FloatTensor,
        sample_weight: torch.FloatTensor,
        detection_importance: float,
        boundbox_importance: float,
        landmark_importance: float
):
    """
    Get loss of the whole network
    0:    face classification (need to +sigmoid -> BCELoss)
    1~4:  bbox
    5~14: landmark
    :param sample_weight: vector size N
    :param y_pred: Nx15 matrix
    :param y_real: Nx15 matrix
    :param detection_importance: weight of face detecation loss
    :param boundbox_importance: weight of bound box regression loss
    :param landmark_importance: weight of landmark regression loss
    :return:
    """
    fc_loss = F.binary_cross_entropy(F.sigmoid(y_pred[:, 0]), target=y_real[:, 0]) * detection_importance
    bb_loss = F.mse_loss(y_pred[:, 1:5], target=y_real[:, 1:5]) * 4 * boundbox_importance
    lm_loss = F.mse_loss(y_pred[:, 5:15], target=y_real[:, 5:15]) * 10 * landmark_importance
    return (fc_loss + bb_loss + lm_loss) * sample_weight


class PNet(nn.Module):
    """
    ProposalNet in the paper
    """

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=15,
                kernel_size=1,
                stride=1
            )
        )
        self.apply(_weights_init)

    def forward(self, x):
        return self.pre_layer(x)


class RNet(nn.Module):
    """
    RefineNet in the paper
    """

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=28,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),
            nn.Conv2d(
                in_channels=28,
                out_channels=48,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=2,
                stride=1
            ),
            nn.PReLU(),
            Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.PReLU(),
            nn.Linear(128, 15)
        )
        self.apply(_weights_init)

    def forward(self, x):
        return self.pre_layer(x)


class ONet(nn.Module):
    """
    ONet in the paper
    """

    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.PReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=1
            ),
            nn.PReLU(),

            Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.PReLU(),
            nn.Linear(128, 15)
        )
        self.apply(_weights_init)

    def forward(self, x):
        return self.pre_layer(x)
