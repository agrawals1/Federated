import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x


class Cifar10FLNet(nn.Module):
    def __init__(self):
        super(Cifar10FLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.fc1 = nn.Linear(4096, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.name = 'cifar10flnet'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)


class CNN_WEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class LeakySoftmaxCNN(nn.Module):
    def __init__(self, input_c, input_h, input_w, kernel_sizes, extra_padding,
                 channel_sizes, final_c=5, activation=None, cuda=False):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0
        self.input_c = input_c
        self.input_h = input_h
        self.input_w = input_w
        self.kernel_sizes = kernel_sizes
        self.extra_padding = extra_padding
        self.channel_sizes = channel_sizes
        self.final_c = final_c
        self.activation = activation
        self.use_cuda = cuda
        self.initialize()

    def initialize(self):
        cnn_layers = []
        c, h, w = self.input_c, self.input_h, self.input_w
        for i, (k, ep) in enumerate(zip(self.kernel_sizes, self.extra_padding)):
            c_in = self.channel_sizes[i-1] if i > 0 else c
            c_out = self.channel_sizes[i]
            p = (k - 1) // 2 + ep
            new_layers = [nn.Conv2d(c_in, c_out, kernel_size=k, padding=p),
                          self.activation(),
                          nn.MaxPool2d(2),
                          nn.Dropout(p=0.0)]
            cnn_layers.extend(new_layers)
            h = h // 2 + ep
            w = w // 2 + ep
        # cnn_layers.append(nn.Conv2d(channel_sizes[-1], final_c,
        #                             kernel_size=1, padding=0))
        self.cnn = nn.Sequential(*cnn_layers)
        self.linear_1 = nn.Linear(h * w * self.channel_sizes[-1], 200)
        self.linear_input_dim = h * w * self.channel_sizes[-1]
        # self.linear_1 = nn.Linear(h * w * final_c, 200)
        # self.linear_input_dim = h * w * final_c
        self.linear_2 = nn.Linear(200, 10)
        self.linear_3 = nn.Linear(10, 1)
        self.bn = nn.BatchNorm1d(10)
        self.double()
        if self.use_cuda:
            self.cuda()

    def forward(self, data):
        # print(data.shape)
        data = F.dropout(self.cnn(data), p=0.0, training=self.training)
        # data = self.cnn(data)
        data = F.dropout(
            F.leaky_relu(self.linear_1(data.view(-1, self.linear_input_dim))),
            p=0.0, training=self.training)

        leaky_class_weights = F.leaky_relu(self.linear_2(data))
        class_probs = F.softmax(self.linear_2(data), dim=1)
        total_weight = leaky_class_weights.sum(1).view(-1, 1)
        data = leaky_class_weights * 0.01 + total_weight * class_probs * 0.99
        data = self.bn(data)
        return self.linear_3(data)

class DefaultCNN(LeakySoftmaxCNN):
    def __init__(self, cuda):
        LeakySoftmaxCNN.__init__(
            self, input_c=1, input_h=28, input_w=28, channel_sizes=[20, 50],
            kernel_sizes=[3, 3], extra_padding=[0, 1],
            activation=nn.LeakyReLU, cuda=cuda)

    def forward(self, data):
        data = data.view(data.shape[0], 1, 28, 28)
        return LeakySoftmaxCNN.forward(self, data)



