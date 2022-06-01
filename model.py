from torch import nn

class NetG(nn.Module):
    def __init__(self,opt):
        super(NetG,self).__init__()
        self.main = nn.Sequential(
            #input is nz noise
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh()
            #state size. 3*96*96
        )
    def forward(self,input):
        return self.main(input)
    
class NetD(nn.Module):
    def __init__(self) -> None:
        super(NetD,self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            #input is (3*96*96)
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #state size. (ndf) x 32 x 32

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            #state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            #state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            #state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
            #state size. 1 x 1 x 1
        )
    def forward(self,input):
        return self.main(input).view(-1)
            
