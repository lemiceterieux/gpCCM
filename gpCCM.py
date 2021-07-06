import numpy as np
import numba as nb
import dcor
import scipy.stats as stats
import scipy.special as sp
import minepy
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class GP():
    def __init__(self):
        self.parameters = None
        self.muMat = None
        self.cuda = 0
        self.A = None
        return

    def setcuda(self, cuda):
        self.cuda = cuda
   
    def testStateSpaceCorrelation(self, X, Y, m=3, tau=1, cuda=0):
        ''' X is the time series to cross map to
            Y is a batch of time series to see how well they crossmap to X
            m is the embedding dimension
            tau is the lag time
            cuda is the GPU to use
        '''
        # Convert to torch tensor if not already a tensor
        if not X is torch.Tensor:
            X = torch.from_numpy(X).float().cuda(cuda)
            Y = torch.from_numpy(np.array(Y)).float().cuda(cuda).T
        # Random map is only defined when we have trained once
        if self.A is None:
            self.A = torch.randn(m,m).cuda(cuda)

        # Standardize
        x = (X - X.mean())/X.std()
        y = (Y - Y.mean(0))/Y.std(0)
        y = y.T
        # State space transform
        xar = torch.stack([x[i:i-tau*m] for i in range(m)]).cuda(cuda).T.matmul(self.A)
        xar = (xar - xar.mean(0))/xar.std(0)
        yar = torch.stack([y[:,i:i-tau*m] for i in range(m)],1).cuda(cuda).transpose(1,2)
        # Train outs
        yp = yar[:,1:,:]
        xp = xar[1:,:]
        # Kernel inputs
        xi = xar[:-1]
        yar = yar[:,:-1]
        # Bayesian Regression
        # set of mus
        mx = []
        kx = []
        mlp = []
    
        self.setcuda(cuda)
    
        for yi, yy in zip(yar,yp):
            # minimize euclidean distance
            ytemp = yi.matmul(torch.inverse(yi.T.matmul(yi)).matmul(yi.T).matmul(xi))
            ml = []
            kl = []
            m, k, mlogp  = self.forward(xi, xp, yi, yy)
            m = m.T
            m[m!=m] = 0
            with torch.no_grad():
                mx.append(m.cpu())
                kx.append(k.cpu())
                mlp.append(mlogp.cpu())
        mx = torch.stack(mx).transpose(0,1).numpy()
        mlp = torch.stack(mlp).transpose(0,1).numpy()
        kx = torch.stack(kx).transpose(0,1)
        return mx, kx.squeeze(), mlp.squeeze(), self

    # Kernels

    def squaredExpKernelARD(self, a, b, ld=3):
        r = torch.zeros(1,a.shape[0], b.shape[0]).cuda(self.cuda)
        for i in range(len(ld)):
            temp = torch.cdist(a[None,:,[i]], b[None,:,[i]])#batch_distance(a[None,:,[i]],b[None,:,[i]],ld[i])
            r += temp**2/ld[i]#**2
        #temp = torch.cdist(a[None,:,:], b[None,:,:])
        #r = temp**2/ld[0]
        cmat = torch.exp(-r)
        return cmat.T.squeeze()

    def optimizeHyperparms(self, data, inp, lr=.001, ld=1., sigma=6., noise=.1, niter=20, ns=False, kernel=None, m=40):
        # Make data not require grad
        data.requires_grad_(False)
        inp.requires_grad_(False)
        kernel = self.squaredExpKernelArt

        train = data

        # Make tensors with gradient
        sigma = torch.tensor(sigma).float().cuda(self.cuda).requires_grad_()
        noiseo = torch.tensor(noise).float().cuda(self.cuda).requires_grad_()
        hnoise = (noise*(torch.ones(m).float())).cuda(self.cuda).requires_grad_()
        noise = torch.tensor(noise).float().cuda(self.cuda).requires_grad_()
        ld = (ld*torch.ones(inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
        p = np.random.permutation(len(data))[:m]
        ipoints = torch.randn(*inp[p].shape).cuda(self.cuda).requires_grad_()


        eye = torch.eye(len(inp)).cuda(self.cuda).float()
        # first postmean evaluation
        # Gradient Ascent: Marginal maximum Log Likelihood p(y|hyperparameters)
        Km = sigma**2*kernel(ipoints, ipoints, ld**2) + (hnoise**2).diag()
        Kn = sigma**2*kernel(inp, inp, ld**2)
        Kmi = torch.inverse(Km)
        Knm = sigma**2*kernel(inp, ipoints, ld**2)
        lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
        K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb +  noiseo**2*eye
        Kinv = torch.inverse(K)
        for i in range(niter):
            logp = 0
            for tr in train.T:
                logp += -.5*tr.T.matmul(Kinv).matmul(tr)
            logp = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()
            #print(logp.item())
            #print(hnoise,noiseo.item(), ld)
            logp.backward()

            with torch.no_grad():
                sigma += lr*sigma.grad
                ld += lr*ld.grad
                noiseo += lr*noiseo.grad
                hnoise += lr*hnoise.grad
                ipoints += lr*ipoints.grad


            sigma.grad.zero_()
            ld.grad.zero_()
            noiseo.grad.zero_()
            hnoise.grad.zero_()
            ipoints.grad.zero_()

            Km = sigma**2*kernel(ipoints, ipoints, ld**2) + (hnoise**2).diag()

            Kn = sigma**2*kernel(inp, inp, ld**2)
            Kmi = torch.inverse(Km)
            Knm = sigma**2*kernel(inp, ipoints, ld**2)
            lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
            K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb +  noiseo**2*eye
            Kinv = torch.inverse(K)

        ld = ld.detach()**2
        sigma = sigma.detach()**2
        noiseo = noiseo.detach()**2
        noise = noise.detach()**2
        self.hnoise = hnoise.detach()**2
        obsparms = [ld, sigma.item(),  noiseo.item(), noise.item(), ipoints]


        for tr in train.T:
            logp += -.5*tr.T.matmul(Kinv).matmul(tr)
        logp = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()
        self.logp = logp.item()
        Qm = Km + Knm.matmul(torch.inverse(lamb + noiseo*eye)).matmul(Knm.T)
        muMatrix = torch.inverse(Qm).matmul(Knm).matmul(torch.inverse(lamb + noiseo*eye)).matmul(train)
        stdMatrix = torch.inverse(Km) - torch.inverse(Qm)
        return logp, obsparms, muMatrix, stdMatrix

    # Posterior Inference
    def forward(self, inp, observations, test, testo, ld=1, sig=10, noise=.1, target="squaredexp"):
        kernel = self.squaredExpKernelARD

        # Get Hyperparms
        if self.muMat is None:
            lp = []
            nsl = []
            noise, sigma, ld = [1,inp.std().item()**(1/2)*2, (inp.std().item())**(1/2)]
            lp, parms, muMatrix, stdMatrix = self.optimizeHyperparms(observations, inp, sigma=sigma, noise=noise, ld=ld ,niter=30,lr=1e-4, kernel=kernel)
            self.sigma = parms[1]
            self.ld = parms[0]
            self.noise = parms[2]
            self.ipoints = parms[-1]
            self.parms = parms
            self.stdMat = stdMatrix
            self.muMat = muMatrix

        ltest = test

        # Test kernel
        tk = self.sigma*kernel(ltest, ltest, self.ld)
            
        # Test cross ipoints 
        inptk = self.sigma*kernel(ltest, self.ipoints, self.ld)
        
        eye = torch.eye(tk.shape[-1]).cuda(self.cuda)

        Km = sigma*kernel(ipoints, ipoints, ld) + (self.hnoise).diag()
        Kmi = torch.inverse(Km)
        Knm = sigma*kernel(ltest, ipoints, ld)
        lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
        K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb +  self.noise*eye
        Kinv = torch.inverse(K)

        marglog = 0
        for tr in train.T:
            marglog += -.5*tr.T.matmul(Kinv).matmul(tr)
        marglog = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()

        # Covariance update
        posteriorK = tk - inptk.T.matmul(self.stdMat).matmul(inptk) + self.noise*eye

        # Predicted test vals
        posteriormu = inptk.T.matmul(self.muMat)
        posteriormu = (posteriormu).T#.T

        return posteriormu, posteriorK, marglog

def testCause(postQuad1, postK1, postQuad2, postK2, margLog1Y, margLog2X):
    '''
        Quad means the quadrature of the exponential in the guassian
        distribution, for example: (mu1 - X1)^TpostK1^-1(mu1 - X1)
        returns the log of the likelihood ratio
    '''
       num = -postQuad1 - torch.linalg.cholesky(postK1).slogdet()[1]
       num = num + margLog2X
       den = -postQuad2 - torch.linalg.cholesky(postK2).slogdet()[1]
       den = den + margLog1X
       return (num - den)/postK1.shape[-1]
