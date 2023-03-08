import torch
import torchvision
from thop import profile
from model.network import deeplabv3plus_resnet50 as S2M

# Model
print('==> Building model..')
net = S2M()
net.load_state_dict(torch.load('saves/s2m.pth', map_location='cpu'))
net = net.eval()
torch.set_grad_enabled(False)

dummy_input = torch.randn(1, 6, 256, 256)
flops, params = profile(net, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
print(net)
