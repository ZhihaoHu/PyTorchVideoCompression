#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs

# gdn = tf.contrib.layers.gdn


class Analysis_net(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        # self.resEncoder = nn.Sequential(
        #     nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),# how to initialize ???
        # )

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


def build_model():
        input_image = Variable(torch.zeros([4, 3, 256, 256]))

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())
        # feature = sess.run(weights)

        # print(weights_val)

        # gamma_val = sess.run(gamma)

        # print(gamma_val)


if __name__ == '__main__':
    build_model()
