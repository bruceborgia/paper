import torch
from torch.utils import data
from MTCNN_Pytorch import simpling
import numpy as np
import os


class Trainer:
    """
        训练网络
    """

    def __init__(self, train_net, batch_size, data_path, save_model_path, lr=0.001, isCuda=True):
        """

        :param train_net: net
        :param batch_size: 批次大小
        :param data_path: 训练集地址
        :param save_model_path: 保存模型地址
        :param lr: 学习率，默认0.0001
        :param isCuda: 是否使用CUDA，默认：TRUE
        """

        self.model = train_net
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.isCuda = isCuda
        self.save_path = save_model_path

        if os.path.exists(self.save_path):  # 如果有保存的模型，加载模型
            self.model = torch.load(self.save_path)

        if self.isCuda:
            self.model.cuda()

        self.face_loss = torch.nn.BCELoss()
        self.offset_loss = torch.nn.MSELoss()

        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.train_net()  # 调用训练方法

    def train_net(self):
        epoch = 1  # 记录训练次数
        IMG_DATA = simpling.FaceDataset(self.data_path)  # 获取数据
        for _ in range(10000):  # 将所有数据训练10000次
            train_data = data.DataLoader(IMG_DATA, batch_size= self.batch_size, shuffle=True, num_workers=4)
            for train in train_data:
                # 获取数据
                # img_data:[512,3,24,24]
                # label : [512,1]
                # offset :[512,4]
                img_data, label, box_offset, land_offset = train

                if self.isCuda:
                    img_data = img_data.cuda()
                    box_offset = box_offset.cuda()
                    land_offset = land_offset.cuda()
                    """
                        获取网络输出：P-NET
                        face_out:[512,2,1,1]
                        box_offset_out:[512,4,1,1]
                        land_offset_out:[512,10,1,1]
                        R-NET,O-NET
                        face_out:[512,2,1,1]
                        box_offset_out:[512,4,1,1]
                        land_offset_out:[512,10,1,1]
                    """
                    face_out, box_offset_out, land_offset_out = self.model(img_data)

                    # 降维 [512, 2, 1, 1] -> [512, 2]
                    face_out = face_out.squeeze()
                    box_offset_out = box_offset_out.squeeze()
                    land_offset_out = land_offset_out.squeeze()

                    # 获取1 和0 做人脸损失
                    one = torch.ne(label, 0)  # two:[512,1]
                    one = one.squeeze()  # one:torch.Size([512]) 掩码输出：1,0,int8

                    # 获取1 和2做回归框损失
                    two = torch.ne(label, 0)  # two:[512,1]
                    two = torch.squeeze()  # two:[512]

                    # 将标签转为one_hot编码
                    label_10 = label[one] # [batch, 1]
                    label_10 = torch.Tensor([self.one_hot(int(i)) for i in label_10.squeeze().numpy()])  # [batch, 2]

                    # 得到人脸损失和偏移量损失
                    face_loss = self.face_loss(face_out[one], label_10.cuda())
                    box_offset_loss = self.offset_loss(box_offset_out[two], box_offset[two])
                    land_offset_loss = self.offset_loss(land_offset_out[two], land_offset[two])

                    # 损失相加
                    self.loss = self.face_loss + box_offset_loss + land_offset_loss

                    #优化损失
                    self.opt.zero_grad()
                    self.loss.backward()
                    self.opt.step()

                    # 每训练100次， 输出损失，并保存数据
                    epoch += 1
                    if epoch % 100 == 0:
                        print('Epoch:', epoch, 'Loss', self.loss.cpu().item())
                        torch.save(self.model, self.save_path)

    def one_hot(self, data):
        """
        one hot 编码
        :param data: 一个值
        :return:onthot编码后的值
        """
        hot = np.zeros([2])
        hot[data] = 1
        return hot

if __name__ == '__main__':
    pass
