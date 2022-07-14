import pdb
from matplotlib.pyplot import flag
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GDropout(nn.Module):
    def __init__(self, p):
        super(GDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True, inplace=False)

class GNet_FCDrop(nn.Module):
    def __init__(self, n_features, n_layers):
        super(GNet_FCDrop, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.net = nn.Sequential(
                nn.Linear(2*n_features, n_features//1), 
                nn.Sigmoid(),
                nn.BatchNorm1d((n_features//1)), 
                GDropout(0.5),
                nn.Linear(n_features//1, n_features//1),
                nn.Sigmoid(),
                nn.BatchNorm1d((n_features//1)), 
                GDropout(0.5),
                nn.Linear(n_features//1, n_features//1),
                nn.Sigmoid(),
                nn.BatchNorm1d((n_features//1)), 
                GDropout(0.5),
                nn.Linear(n_features//1, n_features),
                nn.Tanh()
                )
    def forward(self, x_m_in):
        device = x_m_in.device
        # flatten the input
        x_m = x_m_in.view(-1, self.n_features)
        # get masks, 0 means missing
        mask = 1.0 - torch.isnan(x_m)
        # get a zero imputted input
        x = torch.where(mask.bool(), x_m, torch.tensor(0.0, device=device))
        #z = torch.randn((x_m.shape[0], self.n_features//8)).to(device) 
        # concat mask and x_i
        x = torch.cat([x,mask.float()], dim=1)
        x = self.net(x)
        x_out = x.view(x_m_in.shape)
        # return output
        return x_out

class GNet_FC(nn.Module):
    def __init__(self, n_features, n_layers, layer_factor):
        super(GNet_FC, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_h = int(n_features * layer_factor)
        """
        self.net = nn.Sequential(
                nn.Linear(2*n_features+n_features//8, self.n_h), 
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)),
                nn.Linear(self.n_h, self.n_h),
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)), 
                nn.Linear(self.n_h, self.n_h),
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)), 
                nn.Linear(self.n_h, n_features),
                nn.Tanh()
                )
        """
        self.net = []
        self.net.append(nn.Linear(2*n_features+n_features//8, self.n_h))
        self.net.append(nn.ReLU())
        self.net.append(nn.BatchNorm1d((self.n_h)))
        for _ in range(self.n_layers-1):
            self.net.append(nn.Linear(self.n_h, self.n_h))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm1d((self.n_h)))
        self.net.append(nn.Linear(self.n_h, n_features))
        self.net.append(nn.Tanh())
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x_m_in):
        device = x_m_in.device
        # flatten the input
        x_m = x_m_in.view(-1, self.n_features)
        # get masks, 0 means missing
        mask = 1.0 - torch.isnan(x_m)
        # get a zero imputted input
        x = torch.where(mask.bool(), x_m, torch.tensor(0.0, device=device))
        z = torch.randn((x_m.shape[0], self.n_features//8)).to(device) 
        # concat mask and x_i
        x = torch.cat([x,mask.float(),z], dim=1)
        x = self.net(x)
        x_out = x.view(x_m_in.shape)
        # return output
        return x_out


class DNetF_FC(nn.Module):
    def __init__(self, n_features, n_layers, layer_factor, act_out=nn.Sigmoid()):
        super(DNetF_FC, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_h = int(n_features * layer_factor)
        """
        self.net = nn.Sequential(
                nn.Linear(2*n_features, self.n_h), 
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)), 
                nn.Linear(self.n_h, self.n_h),
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)), 
                nn.Linear(self.n_h, self.n_h),
                nn.ReLU(),
                nn.BatchNorm1d((self.n_h)), 
                nn.Linear(self.n_h, n_features),
                act_out, 
                )
        """
        self.net = []
        self.net.append(nn.Linear(2*n_features, self.n_h))
        self.net.append(nn.ReLU())
        self.net.append(nn.BatchNorm1d((self.n_h)))
        for _ in range(self.n_layers-1):
            self.net.append(nn.Linear(self.n_h, self.n_h))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm1d((self.n_h)))
        self.net.append(nn.Linear(self.n_h, n_features))
        self.net.append(act_out)
        
        self.net = nn.Sequential(*self.net)


    def forward(self, x_m_in):
        # flatten the input
        x_out = self.net(x_m_in)
        # return output
        return x_out


class GNet_FCRes(nn.Module):
    def __init__(self, n_features, n_blocks, n_hidden):
        super(GNet_FCRes, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_blocks = n_blocks
        # input layer
        self.n_h = n_hidden
        #self.layer_in = torch.nn.Linear(2*self.n_features+64, n_layers[0])
        #self.bnorm_in = torch.nn.BatchNorm1d(n_h)
        #self.layer_in2 = torch.nn.Linear(n_h, n_h)
        #self.bnorm_in2 = torch.nn.BatchNorm1d(n_h)
        # hidden layers
        self.layer_hiddens = torch.nn.ModuleList()
        self.layer_hiddens.append(InBlockFC(2*self.n_features+self.n_features//8, self.n_h))
        self.layer_hiddens.append(nn.BatchNorm1d((self.n_h)))
        n_last = self.n_h
        for ind_blk in range(self.n_blocks):
            self.layer_hiddens.append(ResBlockFC(n_last))
            # TODO: add dim reduction logic
        # output layer
        self.layer_out = torch.nn.Linear(n_last, self.n_features)
        self.act_fn = torch.nn.ReLU(inplace=True)
        self.act_out = torch.nn.Tanh()
    
    def forward(self, x_m_in):
        device = x_m_in.device
        # flatten the input
        x_m = x_m_in.view(-1, self.n_features)
        # get masks, 0 means missing
        mask = 1.0 - torch.isnan(x_m)
        # get zero imputted input
        x_i = torch.where(mask.bool(), x_m, torch.tensor(0.0,device=device))
        # concat mask and x_i
        z = torch.randn((x_i.shape[0], self.n_features//8)).to(device) #TODO: try different z sizes
        x = torch.cat([x_i,mask.float(),z], dim=1)
        
        # input layer
        #x = self.bnorm_in(self.act_fn(self.layer_in(x)))
        #x = self.bnorm_in2(self.act_fn(self.layer_in2(x)))
        # for all layers except the last one
        for layer in self.layer_hiddens:
            x = layer(x)
        # last layer
        x = self.act_out(self.layer_out(x))
        x_out = x.view(x_m_in.shape)
        # return output
        return x_out


class DNetF_FCRes(nn.Module):
    def __init__(self, n_features, n_blocks, n_hidden):
        super(DNetF_FCRes, self).__init__()
        # set attributes
        self.n_features = n_features
        self.n_blocks = n_blocks
        # input layer
        self.n_h = n_hidden
        #self.layer_in = torch.nn.Linear(self.n_features, n_layers[0])
        # hidden layers
        self.layer_hiddens = torch.nn.ModuleList()
        self.layer_hiddens.append(InBlockFC(self.n_features*2, self.n_h))
        self.layer_hiddens.append(nn.BatchNorm1d((self.n_h)))
        n_last = self.n_h
        for ind_blk in range(self.n_blocks):
            self.layer_hiddens.append(ResBlockFC(n_last))
            # TODO: add dim reduction logic
        # output layer
        self.layer_out = torch.nn.Linear(n_last, self.n_features)
        self.act_out = torch.nn.Sigmoid()

    def forward(self, x_in):
        # flatten the input
        #x = x_in.view(-1, self.n_features)
        # input layer
        #x = self.layer_in(x)
        # for all layers except the last one
        x = x_in
        for layer in self.layer_hiddens:
            x = layer(x)
        # last layer
        x_out = self.act_out(self.layer_out(x))
        #x_out = x.view(x_in.shape)
        # return output
        return x_out



class ResBlockFC(nn.Module):
    def __init__(self, n_hidden):
        super(ResBlockFC, self).__init__()
        # attributes
        self.n_h = n_hidden
        # layers
        self.fc1 = nn.utils.spectral_norm(torch.nn.Linear(self.n_h, self.n_h))
        self.bn1 = torch.nn.BatchNorm1d((self.n_h))
        self.fc2 = nn.utils.spectral_norm(torch.nn.Linear(self.n_h, self.n_h))
        self.bn2 = torch.nn.BatchNorm1d((self.n_h))
        self.act_fn = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, self.n_h)
        # fc 1
        out = self.fc1(x)
        out = self.act_fn(out)
        out = self.bn1(out)
        # fc 2
        out = self.fc2(out)
        # res connection
        out += x
        out = self.act_fn(out)
        out = self.bn2(out)
        return out

class InBlockFC(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(InBlockFC, self).__init__()
        # attributes
        self.n_h = n_hidden
        self.n_features = n_features
        # layers
        self.fc1 = nn.utils.spectral_norm(torch.nn.Linear(self.n_features, self.n_h))
        self.bn1 = torch.nn.BatchNorm1d((self.n_h))
        self.fc2 = nn.utils.spectral_norm(torch.nn.Linear(self.n_h, self.n_h))
        self.bn2 = torch.nn.BatchNorm1d((self.n_h))
        self.act_fn = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, self.n_features)
        # fc 1
        out = self.fc1(x)
        out = self.act_fn(out)
        out = self.bn1(out)
        # fc 2
        out = self.fc2(out)
        out = self.act_fn(out)
        out = self.bn2(out)
        return out


class ConvBlock_ResNet(nn.Module):
    def __init__(self, n_channels):
        super(ConvBlock_ResNet, self).__init__()
        self.n_channels = n_channels
        self.conv = nn.Sequential(
            #nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, 3, padding=1)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            #nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, 3, padding=1)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_out = self.conv(x) + x
        return x_out
        

class ConvBlock_UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock_UNet, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            #nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            #nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = self.conv(torch.cat([x1,x2], dim=1))
        else:
            x = self.conv(x1)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        self.query = nn.utils.spectral_norm(
                nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = nn.utils.spectral_norm(
                nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = nn.utils.spectral_norm(
                nn.Conv1d(in_channel, in_channel, 1))

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + x

        return out


class GNet_ResNet(nn.Module):
    def __init__(self, in_channels=3, n_downsampling=2, n_blocks=4, attention=False):
        super(GNet_ResNet, self).__init__()
        ngf = 64 
        self.net = []
        # input layers
        self.net += [nn.utils.spectral_norm(nn.Conv2d(in_channels+2, ngf, kernel_size=7, padding=3))]
        self.net += [nn.BatchNorm2d(ngf)]
        self.net += [nn.ReLU(inplace=True)]
        for i in range(n_downsampling):
            self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf*(2**i), ngf*(2**(1+i)), 
                kernel_size=3, padding=1, stride=2))]
            self.net += [nn.BatchNorm2d(ngf*(2**(1+i)))]
            self.net += [nn.ReLU(inplace=True)]
        # self-attention
        if attention:
            self.net += [SelfAttention(ngf*(2**n_downsampling))]
        # res blocks
        for i in range(n_blocks):
            self.net += [ConvBlock_ResNet(ngf*(2**n_downsampling))]
        # output layers
        for i in range(n_downsampling,0,-1):
            self.net += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*(2**i), ngf*(2**(i-1)), 
                kernel_size=3, padding=1, stride=2, output_padding=1))]
            self.net += [nn.BatchNorm2d(ngf*(2**(i-1)))]
            self.net += [nn.ReLU(inplace=True)]
        self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf, in_channels, kernel_size=7, padding=3))]
        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x_m):
        device = x_m.device
        # get masks, 0 means missing
        mask = 1.0 - torch.isnan(x_m).float()
        # get zero imputted input
        x_i = torch.where(mask.bool(), x_m, torch.tensor(0.0,device=device))
        mask = torch.unsqueeze(mask[:,0,:,:], 1)
        # concat mask and x_i
        z = torch.randn(mask.shape, device=device) * (1.0-mask.float())
        x = torch.cat([x_i,mask.float(),z], dim=1)
        # forward path
        x_out = self.net(x)
        return x_out

class DNetF_ResNet(nn.Module):
    def __init__(self, in_channels=3, n_downsampling=2, n_blocks=4, 
            attention=False, act_out=nn.Sigmoid()):
        super(DNetF_ResNet, self).__init__()
        ngf = 64  
        self.net = []
        # input layers
        self.net += [nn.utils.spectral_norm(nn.Conv2d(in_channels+1, ngf, kernel_size=7, padding=3))]
        self.net += [nn.BatchNorm2d(ngf)]
        self.net += [nn.ReLU(inplace=True)]
        for i in range(n_downsampling):
            self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf*(2**i), ngf*(2**(1+i)), 
                kernel_size=3, padding=1, stride=2))]
            self.net += [nn.BatchNorm2d(ngf*(2**(1+i)))]
            self.net += [nn.ReLU(inplace=True)]
        # self-attention
        if attention:
            self.net += [SelfAttention(ngf*(2**n_downsampling))]
        # res blocks
        for i in range(n_blocks):
            self.net += [ConvBlock_ResNet(ngf*(2**n_downsampling))]
        # output layers
        for i in range(n_downsampling,0,-1):
            self.net += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*(2**i), ngf*(2**(i-1)), 
                kernel_size=3, padding=1, stride=2, output_padding=1))]
            self.net += [nn.BatchNorm2d(ngf*(2**(i-1)))]
            self.net += [nn.ReLU(inplace=True)]
        self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf, in_channels, kernel_size=7, padding=3))]
        if act_out != None:
            self.net += [act_out]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        # forward path
        x_out = self.net(x)
        return x_out
    

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, 
                mode=self.mode, align_corners=False)
        return x

class GNet_AttnNet(nn.Module):
    def __init__(self, in_channels=3, n_downsampling=2, n_blocks=4):
        super(GNet_AttnNet, self).__init__()
        ngf = 64 
        self.net = []
        # input layers
        self.net += [nn.utils.spectral_norm(nn.Conv2d(in_channels+2, ngf, kernel_size=7, padding=3))]
        self.net += [nn.BatchNorm2d(ngf)]
        self.net += [nn.ReLU(inplace=True)]
        for i in range(n_downsampling):
            self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf*(2**i), ngf*(2**(1+i)), 
                kernel_size=3, padding=1, stride=2))]
            self.net += [nn.BatchNorm2d(ngf*(2**(1+i)))]
            self.net += [nn.ReLU(inplace=True)]
        # attn blocks
        for i in range(n_blocks):
            self.net += [SelfAttention(ngf*(2**n_downsampling)),
                            ConvBlock_ResNet(ngf*(2**n_downsampling))]
        # output layers
        for i in range(n_downsampling,0,-1):
            self.net += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*(2**i), ngf*(2**(i-1)), 
                kernel_size=3, padding=1, stride=2, output_padding=1))]
            self.net += [nn.BatchNorm2d(ngf*(2**(i-1)))]
            self.net += [nn.ReLU(inplace=True)]
        self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf, in_channels, kernel_size=7, padding=3))]
        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x_m):
        device = x_m.device
        # get masks, 0 means missing
        mask = 1.0 - torch.tensor(torch.isnan(x_m), dtype=float)
        # get zero imputted input
        x_i = torch.where(mask.bool(), x_m, torch.tensor(0.0,device=device))
        mask = torch.unsqueeze(mask[:,0,:,:], 1)
        # concat mask and x_i
        z = torch.randn(mask.shape, device=device) * (1.0-mask.float())
        x = torch.cat([x_i,mask.float(),z], dim=1)
        # forward path
        x_out = self.net(x)
        return x_out

class DNetF_AttnNet(nn.Module):
    def __init__(self, in_channels=3, n_downsampling=2, n_blocks=4, act_out=nn.Sigmoid()):
        super(DNetF_AttnNet, self).__init__()
        ngf = 64  
        self.net = []
        # input layers
        self.net += [nn.utils.spectral_norm(nn.Conv2d(in_channels+1, ngf, kernel_size=7, padding=3))]
        self.net += [nn.BatchNorm2d(ngf)]
        self.net += [nn.ReLU(inplace=True)]
        for i in range(n_downsampling):
            self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf*(2**i), ngf*(2**(1+i)), 
                kernel_size=3, padding=1, stride=2))]
            self.net += [nn.BatchNorm2d(ngf*(2**(1+i)))]
            self.net += [nn.ReLU(inplace=True)]
        # attn blocks
        for i in range(n_blocks):
            self.net += [SelfAttention(ngf*(2**n_downsampling)),
                            ConvBlock_ResNet(ngf*(2**n_downsampling))]
        # output layers
        for i in range(n_downsampling,0,-1):
            self.net += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*(2**i), ngf*(2**(i-1)), 
                kernel_size=3, padding=1, stride=2, output_padding=1))]
            self.net += [nn.BatchNorm2d(ngf*(2**(i-1)))]
            self.net += [nn.ReLU(inplace=True)]
        self.net += [nn.utils.spectral_norm(nn.Conv2d(ngf, in_channels, kernel_size=7, padding=3))]
        if act_out != None:
            self.net += [act_out]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        # forward path
        x_out = self.net(x)
        return x_out

class GNet_UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(GNet_UNet, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.up = Interpolate(scale_factor=2, mode='bilinear')
        self.convblock_l1 = ConvBlock_UNet(in_channels+2, 64)
        self.convblock_l2 = ConvBlock_UNet(64, 128)
        self.convblock_l3 = ConvBlock_UNet(128, 256)
        self.convblock_l4 = ConvBlock_UNet(256, 512)
        self.convblock_5 = ConvBlock_UNet(512, 512)
        self.convblock_r4 = ConvBlock_UNet(1024, 256)
        self.convblock_r3 = ConvBlock_UNet(512, 128)
        self.convblock_r2 = ConvBlock_UNet(256, 64)
        self.convblock_r1 = ConvBlock_UNet(128, 64) 
        self.convout = nn.Sequential(
                nn.Conv2d(64, in_channels, 1, padding=0), 
                nn.Tanh())

    def forward(self, x_m):
        device = x_m.device
        # get masks, 0 means missing
        mask = 1.0 - torch.isnan(x_m)
        # get zero imputted input
        x_i = torch.where(mask.bool(), x_m, torch.tensor(0.0,device=device))
        mask = torch.unsqueeze(mask[:,0,:,:], 1)
        # concat mask and x_i
        z = torch.randn(mask.shape, device=device) * (1.0-mask.float())
        x = torch.cat([x_i,mask.float(),z], dim=1)
        # forward path
        x1 = self.convblock_l1(x)
        x2 = self.convblock_l2(self.down(x1))
        x3 = self.convblock_l3(self.down(x2))
        x4 = self.convblock_l4(self.down(x3))
        x5 = self.convblock_5(self.down(x4))
        x = self.convblock_r4(self.up(x5), x4)
        x = self.convblock_r3(self.up(x), x3)
        x = self.convblock_r2(self.up(x), x2)
        x = self.convblock_r1(self.up(x), x1)
        x = self.convout(x)
        return x


class DNetF_UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(DNetF_UNet, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.up = Interpolate(scale_factor=2, mode='bilinear')
        self.convblock_l1 = ConvBlock_UNet(in_channels+1, 64)
        self.convblock_l2 = ConvBlock_UNet(64, 128)
        self.convblock_l3 = ConvBlock_UNet(128, 256)
        self.convblock_l4 = ConvBlock_UNet(256, 512)
        self.convblock_5 = ConvBlock_UNet(512, 512)
        self.convblock_r4 = ConvBlock_UNet(1024, 256)
        self.convblock_r3 = ConvBlock_UNet(512, 128)
        self.convblock_r2 = ConvBlock_UNet(256, 64)
        self.convblock_r1 = ConvBlock_UNet(128, 64) 
        self.convout = nn.Sequential(
                nn.Conv2d(64, 3, 1, padding=0), 
                nn.Sigmoid())

    def forward(self, x_i):
        # forward path
        x1 = self.convblock_l1(x_i)
        x2 = self.convblock_l2(self.down(x1))
        x3 = self.convblock_l3(self.down(x2))
        x4 = self.convblock_l4(self.down(x3))
        x5 = self.convblock_5(self.down(x4))
        x = self.convblock_r4(self.up(x5), x4)
        x = self.convblock_r3(self.up(x), x3)
        x = self.convblock_r2(self.up(x), x2)
        x = self.convblock_r1(self.up(x), x1)
        x = self.convout(x)
        return x


class DNetV_ResNet(nn.Module):
    def __init__(self, block=torchvision.models.resnet.BasicBlock, 
            layers=[2,2,2,2], num_classes=1, zero_init_residual=False):
        super(DNetV_ResNet, self).__init__()
        self.inplanes = 64
        # 3 channels for image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # output act
        self.act_out = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.utils.spectral_norm(nn.Conv2d(self.inplanes, planes * block.expansion, 
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, m_in):
        #device = x_in.device
        # get zero imputted input
        #x = torch.where(mask.byte(), x_in, torch.tensor(0.0,device=device))
        #mask = torch.unsqueeze(mask[:,0,:,:], 1)
        # concat mask and x_i
        #x = torch.cat([x,mask.float()], dim=1)
        x = m_in
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.act_out(self.fc(x))
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

