import torch
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


a = 1.0
b = 0.0
c = 0.0

def f(x):
    return a*torch.pow(x,2.0) + b*x + c

x = torch.linspace(-5,5,100)
y = a*torch.pow(x,2.0)+b*c

plt.figure()
plt.plot(x.numpy(),y.numpy())
plt.show()
t0 = 4*torch.ones(1)

lf = f(t0)
t = t0
t.requires_grad = True
optimizer = torch.optim.SGD([t],lr = 0.1)
i = 0
path = []
path.append([t0.item(),lf.item()])

while lf.item() > 1e-2:
    optimizer.zero_grad()
    lf = f(t)
    lf.backward()
    print(i,lf.item())
    optimizer.step()
    xp = path[-1].copy()
    lf = f(t)
    path.append([t.item(),lf.item()])

    # function to add arrow on a graph
    x1 = xp[0]
    y1 = xp[1]
    x2 = t.item()
    y2 = lf.item()
    plt.arrow(x1,y1,x2-x1,y2-y1, width=0.1,color='red')
    i = i+1

qq = 0



y