import simulateProcesses as sp
import numpy as np
import vgpCCM as gp
import matplotlib 
matplotlib.use("agg")
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
import torch.multiprocessing as mp


def tester(x):
    print("Entering")
    cuda = (mp.current_process()._identity[0] - 1)%2
    coup = x[0]
    x = x[1]
    ret = []
    ret2 = []
    for i in range(3):
        GP = gp.GP()
        GP2 = gp.GP()
        ret += [GP.testStateSpaceCorrelation(x[i], x[3:], 13, tau=2, cuda=cuda)[1].cpu().numpy()]
        ret2 += [GP2.testStateSpaceCorrelation(x[i+3], x[:3], 13, tau=2, cuda=cuda)[1].cpu().numpy()]

    ret = np.array(ret)
    ret2 = np.array(ret2)
    print(ret.shape,ret2.shape,coup)
    res = [[] for i in range(3)]
    for i in range(3):
        for j in range(3):
            res[i] += [np.tanh(1/3000*(ret[i,j][:,None] - ret2[j,i]).ravel())]
    print(coup,np.array(res).shape)
    return np.array(res)

res = []
for i in range(30):
    x0 = np.array(sp.lorenzDrivesRossler(N=3000,dnoise=1e-5, h=.1, eps=0, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
    x1 = np.array(sp.lorenzDrivesRossler(N=3000,dnoise=1e-5, h=.1, eps=2, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
    x3 = np.array(sp.lorenzDrivesRossler(N=3000,dnoise=1e-5, h=.1, eps=4, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
    x6 = np.array(sp.lorenzDrivesRossler(N=3000,dnoise=1e-5, h=.1, eps=6, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
    
    with Pool(2) as p:
        res += [p.map(tester,[[0,x0],[1,x1],[3,x3],[6,x6]])]
res = np.array(res)
np.save("ResLor.npy",res)
label = [[] for i in range(3)]
for i in range(3):
    for j in range(3):
        label[i] += ["X{0:d}-Y{1:d}".format(i,j)]

for i,r in enumerate(res[0]):
    for j in range(len(r)):
        for k in range(len(r[0])):
            rr = r[j,k]
            rasort = np.argsort(rr)
            CDF = np.arange(len(rasort))/len(rasort)
            plt.plot(rr[rasort],CDF)
            plt.xlim((-1,1))
            plt.axvline(rr[0],color="red")
            plt.xlabel(r"$\kappa$")
            plt.ylabel(r"$P(\tilde \kappa)$")
            plt.title(r"Coupling $\epsilon = {0:.2f}$ pval={1:.3f}".format([0,2,4,6][i],1-CDF[rasort==0].squeeze()))
            plt.savefig("Coupling"+str([0,2,4,6][i])+"_"+label[j][k]+".png")
            plt.close()
        
