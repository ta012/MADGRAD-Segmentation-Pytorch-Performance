#!/home/tony/anaconda3/envs/pytorch17_102/bin/python

"""
To do
Xavier init #### done
bias in decoder conv2d
Remove Maxpool2d and use stride
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn as nn
from torch.nn import Conv2d,BatchNorm2d,ReLU,MaxPool2d


import time


def _xavier_init_(m):
    if isinstance(m, nn.Conv2d):
        # print(m)
        nn.init.xavier_uniform_((m.weight))
        # print(m.bias)
        if m.bias is not None:## bias is None in renet backbone, because batchnorm follows it
            # nn.init.xavier_uniform_((m.bias))
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # print(m)
        nn.init.constant_(m.weight, 1) # alpha in doc equation 
        nn.init.constant_(m.bias, 0) # beta in doc equation

def _kaiming_init_(m):

    if isinstance(m, nn.Conv2d):
        # print(m)
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # print(m.bias)
        if m.bias is not None:## bias is None in renet backbone, because batchnorm follows it
            # nn.init.xavier_uniform_((m.bias))
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # print(m)
        nn.init.constant_(m.weight, 1) # alpha in doc equation 
        nn.init.constant_(m.bias, 0) # beta in doc equation
        

def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False), ## bias used because no batchnorm after this
        BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ReLU(inplace=True),
    )
class CustomUNet(nn.Module):

    def __init__(self, n_class,two_pow):
        super().__init__()
         

        temp1 = [Conv2d(3, int(2**two_pow), kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                ,BatchNorm2d(int(2**two_pow), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)]            
        temp2 = [MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),Conv2d(int(2**two_pow), int(2**two_pow), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**two_pow), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**two_pow), int(2**two_pow), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**two_pow), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**two_pow), int(2**two_pow), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**two_pow), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**two_pow), int(2**two_pow), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**two_pow), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)

              ]
        temp3 = [
                Conv2d(int(2**two_pow), int(2**(two_pow+1)), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+1)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+1)), int(2**(two_pow+1)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+1)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)

                ,Conv2d(int(2**(two_pow+1)), int(2**(two_pow+1)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+1)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+1)), int(2**(two_pow+1)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+1)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
            ]
        temp4 = [Conv2d(int(2**(two_pow+1)), int(2**(two_pow+2)), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+2)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+2)), int(2**(two_pow+2)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+2)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+2)), int(2**(two_pow+2)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+2)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+2)), int(2**(two_pow+2)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+2)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
            ]

        temp5 = [Conv2d(int(2**(two_pow+2)), int(2**(two_pow+3)), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+3)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+3)), int(2**(two_pow+3)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+3)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+3)), int(2**(two_pow+3)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+3)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
                ,Conv2d(int(2**(two_pow+3)), int(2**(two_pow+3)), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                ,BatchNorm2d(int(2**(two_pow+3)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ,ReLU(inplace=True)
            ]




            
        self.layer0 = nn.Sequential(*temp1) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convbnrelu(int(2**two_pow), int(2**two_pow), 1, 0)
        self.layer1 = nn.Sequential(*temp2) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convbnrelu(int(2**two_pow), int(2**two_pow), 1, 0)       
        self.layer2 = nn.Sequential(*temp3)   # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convbnrelu(int(2**(two_pow+1)),int(2**(two_pow+1)), 1, 0)  
        self.layer3 = nn.Sequential(*temp4)  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convbnrelu(int(2**(two_pow+2)),int(2**(two_pow+2)), 1, 0)  
        self.layer4 = nn.Sequential(*temp5)  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convbnrelu(int(2**(two_pow+3)), int(2**(two_pow+3)), 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # self.conv_up3 = convbnrelu(int(2**(two_pow+3)), int(2**(two_pow+1)), 3, 1)
        # self.conv_up2 = convbnrelu(int(2**(two_pow+1)), int(2**(two_pow)), 3, 1)
        # self.conv_up1 = convbnrelu(int(2**(two_pow)), int(2**(two_pow-1)), 3, 1)
        # self.conv_up0 = convbnrelu(int(2**(two_pow-1)), int(2**(two_pow-1)), 3, 1)

        # #######
        self.conv_up3 = convbnrelu(int(2**(two_pow+3))+int(2**(two_pow+2)), int(2**(two_pow+1)), 3, 1)
        self.conv_up2 = convbnrelu(int(2**(two_pow+1))+int(2**(two_pow+1)), int(2**(two_pow)), 3, 1)
        self.conv_up1 = convbnrelu(int(2**(two_pow))+int(2**two_pow), int(2**(two_pow-1)), 3, 1)
        self.conv_up0 = convbnrelu(int(2**(two_pow-1))+int(2**two_pow), int(2**(two_pow-1)), 3, 1)
        # ##########
        
        self.conv_original_size0 = convbnrelu(3, int(2**two_pow), 3, 1)
        self.conv_original_size1 = convbnrelu(int(2**two_pow), int(2**(two_pow-1)), 3, 1)
        self.conv_original_size2 = convbnrelu(int(2**(two_pow-1))+int(2**(two_pow-1)), int(2**(two_pow-1)), 3, 1)
        
        self.conv_last = nn.Conv2d(int(2**(two_pow-1)), n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        # layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)

        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)

        x = self.conv_up2(x)


        x = self.upsample(x)

        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)

        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out


    def _initialize_(self,):
        self.apply(_xavier_init_)

    def _kaiming_initialize_(self,):
        self.apply(_kaiming_init_)

if __name__ == "__main__":
    from torchsummary import summary



    """
    torch.Size([1, 256, 32, 32])
    torch.Size([1, 128, 64, 64])
    torch.Size([1, 256, 64, 64])
    torch.Size([1, 256, 128, 128])
    torch.Size([1, 128, 256, 256])
    torch.Size([1, 64, 512, 512])


    """
    # device = 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    num_class = 1
    rep = 10
    resolution = 320
    two_pow = 4
    # k=4

    model =  CustomUNet(num_class,two_pow).to(device)
    print(model)

    model._kaiming_initialize_()
    model.eval()
    print(summary(model,(3,resolution,resolution)))
    model = torch.jit.script(model)
    torch.jit.save(model,'test.pt')
    model.eval()

    inp = torch.ones(1,3,resolution,resolution).to(device)
    out = model(inp)
    inp = torch.ones(1,3,resolution,resolution).to(device)
    out = model(inp)
    print(out.shape)
    # exit()
    o = out.detach().cpu()
    time_l = []
    for i in range(rep):
        inp = torch.ones(1,3,resolution,resolution).to(device)

        start_time = time.time()
        out = model(inp)
        o = out.detach().cpu()
        # torch.cuda.synchronize()

        time_l.append(time.time()-start_time)

        print("out")
       
    
    print(sum(time_l)/len(time_l))
    print(time_l)



