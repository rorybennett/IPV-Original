from torch import nn, hub, abs, div
import torch
import parameters as pms
from skimage import io

class Nothing(torch.nn.Module):
    def __init__(self):
        super(Nothing, self).__init__()
    def forward(self, x):
        return x

class Quadruplet(nn.Module):

    def __init__(self):
        super(Quadruplet, self).__init__()

        self.net1 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True).cuda()
        # self.net1.avgpool = Nothing()
        # self.net1.fc = Nothing()
        self.net1.fc = torch.nn.Linear(512, 128)

        self.net2 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True).cuda()
        # self.net2.avgpool = Nothing()
        # self.net2.fc = Nothing()
        self.net2.fc = torch.nn.Linear(512, 128)

        self.net3 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True).cuda()
        # self.net3.avgpool = Nothing()
        # self.net3.fc = Nothing()
        self.net3.fc = torch.nn.Linear(512, 128)

        self.net4 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True).cuda()
        # self.net4.avgpool = Nothing()
        # self.net4.fc = Nothing()
        self.net4.fc = torch.nn.Linear(512, 128)

        self.l1 = torch.nn.Linear(512, pms.num_of_classes[0])
        self.l2 = torch.nn.Linear(512, pms.num_of_classes[1])
        self.l3 = torch.nn.Linear(512, pms.num_of_classes[0])
        self.l4 = torch.nn.Linear(512, pms.num_of_classes[1])
        if pms.num_of_pts == 4:
            self.l5 = torch.nn.Linear(512, pms.num_of_classes[0])
            self.l6 = torch.nn.Linear(512, pms.num_of_classes[1])
            self.l7 = torch.nn.Linear(512, pms.num_of_classes[0])
            self.l8 = torch.nn.Linear(512, pms.num_of_classes[1])


    def forward(self, x):
        # img = torch.cat((x[:, 0][0][0], x[:, 1][0][0], x[:, 2][0][0], x[:, 3][0][0]))
        # io.imshow(img.detach().cpu().numpy())
        # io.show()

        net1_out = self.net1(x[:, 0])
        net2_out = self.net2(x[:, 1])
        net3_out = self.net3(x[:, 2])
        net4_out = self.net4(x[:, 3])

        net_out = torch.cat((net1_out, net2_out, net3_out, net4_out), 1)

        x1 = self.l1(net_out)
        x2 = self.l2(net_out)
        x3 = self.l3(net_out)
        x4 = self.l4(net_out)
        if pms.num_of_pts == 4:
            x5 = self.l5(net_out)
            x6 = self.l6(net_out)
            x7 = self.l7(net_out)
            x8 = self.l8(net_out)

        if pms.num_of_pts == 2:
            return x1, x2, x3, x4
        else:
            return x1, x2, x3, x4, x5, x6, x7, x8