import numpy as np
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def CausalStat(K1, K2):
    # From the derivation of K from DELTA posterior entropy of same dimensional
    # covariances
    return torch.log(K1) - torch.log(K2)

class GP():
    def __init__(self):
        self.trained = False
        return

    def setcuda(self, cuda):
        self.cuda = cuda
   
    def process(self, X, Y, m=3, tau=1, cuda=0):
        if not X is torch.Tensor:
            X = torch.from_numpy(X).float().cuda(cuda)
            Y = torch.from_numpy(np.array(Y)).float().cuda(cuda).T

        # Standardize
        x = (X - X.mean())/X.std()
        y = (Y - Y.mean(0))/Y.std(0)
        y = y.T
        # State space transform
        xar = torch.stack([x[i:i-tau*m] for i in range(m)]).cuda(cuda).T
        xar = (xar - xar.mean(0))/xar.std(0)
        yar = torch.stack([y[:,i:i-tau*m] for i in range(m)],1).cuda(cuda).transpose(1,2)

        # Train outs
        yp = yar[:,1:,:]
        xp = xar[1:,:]
        xi = xar[:-1]
        yar = yar[:,:-1]

        # Bayesian Regression
        mx = []
        kx = []
    
    
        for yi, yy in zip(yar,yp):
            m, k = gpx.forward(xi, xp, yi, yy)
            mx += [m]
            kx += [k]
        return mx, kx

    def squaredExpKernelARD(self, a, b, ld=3):
        r = torch.zeros(1,a.shape[0], b.shape[0]).cuda(self.cuda)
        for i in range(len(ld)):
            temp = torch.cdist(a[None,:,[i]], b[None,:,[i]])
            r += temp**2/ld[i]
        cmat = torch.exp(-r)
        return cmat.T.squeeze()

    def optimizeHyperparms(self, data, inp, lr=.001, ld=1., sigma=6., noise=.1, niter=20, ns=False, kernel=None, m=40):
        data.requires_grad_(False)
        inp.requires_grad_(False)
        if kernel is None:
            kernel = self.squaredExponentialARD
        data = (data)
        ip = inp
        train = data

        # Define Hyperparameters and their distributions
        A = (torch.randn(inp.shape[-1],inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
        V = torch.linalg.svd(A)[-1]
        sigma = torch.tensor(1*sigma).float().log().cuda(self.cuda).requires_grad_()
        noiseo = torch.tensor(noise).float().cuda(self.cuda).requires_grad_()
        hnoise = (noise*(torch.ones(m).float())).cuda(self.cuda).requires_grad_()
        sld = (1*torch.ones(inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
        ld = (ld*torch.ones(inp.shape[-1])).float().log().cuda(self.cuda).requires_grad_()
        ssigma = (1*torch.ones(1)).float().cuda(self.cuda).requires_grad_()

        # Get some nice inducing points
        p = np.random.permutation(len(data))[:m]
        inp = ip.matmul(V)
        sipoints = (1*torch.ones(*inp[p].shape)).cuda(self.cuda).requires_grad_()
        ipoints = torch.clone(inp[p].detach()).cuda(self.cuda).requires_grad_()

        msigma = sigma.data.requires_grad_(False)
        mipoints = ipoints.data.requires_grad_(False)
        mld = ld.data.requires_grad_(False)

        # put hyperparms we want to gradclip in list
        parms = [sigma,ssigma,ld,sld,ipoints,sipoints]
        
        mSamples =5
        eye = torch.eye(len(inp)).cuda(self.cuda).float()
        for i in range(niter):
            ls = []
            for k in range(mSamples):
                # Sample hyperparameters
                V = torch.linalg.svd(A)[-1]
                inp = ip.matmul(V)
                ripoints = (ipoints+(sipoints*torch.randn(*ipoints.shape).cuda(self.cuda)))
                rld = (ld+(sld*torch.randn(*ld.shape).cuda(self.cuda))).exp()
                rsigma = (sigma+(ssigma*torch.randn(1).cuda(self.cuda))).exp()
                # Sparse kernel GP likelihood log likelihood
                Km = rsigma*kernel(ripoints, ripoints, rld) + (hnoise**2).diag()
                Kn = rsigma*kernel(inp, inp, rld)
                Kmi = torch.inverse(Km)
                Knm = rsigma*kernel(inp, ripoints, rld)
                lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
                K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb +  noiseo**2*eye
                Kinv = torch.inverse(K)
                logp = 0
                trr = train.matmul(V)
                for tr in trr.T:
                    logp += -.5*tr.T.matmul(Kinv).matmul(tr)
                logp = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()
                (logp/mSamples).backward()
                ls += [logp.item()]
            # KL Divergence between approximate posterior and prior distribution
            logp =  -1/2*((-msigma + sigma)**2 -1 + ssigma**2) - torch.log(ssigma/1)
            logp -= (1/2*((-mld + ld)**2 -1 + sld**2) - torch.log(sld/1)).sum()
            logp -= (1/2*((-mipoints + ipoints)**2 -1 + sipoints**2) - torch.log(sipoints/1)).sum()

            # Autograd
            logp.backward()
            # Grad clip
            torch.nn.utils.clip_grad_norm_(parms,1000)

            # Gradient Ascent
            with torch.no_grad():
                sigma += lr*sigma.grad
                ld += lr*ld.grad
                noiseo += lr*noiseo.grad
                hnoise += lr*hnoise.grad
                ipoints += lr*ipoints.grad
                ssigma += lr*ssigma.grad
                sld += lr*sld.grad
                sipoints += lr*sipoints.grad
                A += lr*A.grad

            # Zero out calculated grad
            sigma.grad.zero_()
            ld.grad.zero_()
            noiseo.grad.zero_()
            hnoise.grad.zero_()
            ipoints.grad.zero_()
            A.grad.zero_()

        
        self.ld = ld.detach().requires_grad_(False)
        self.sld = sld.detach().requires_grad_(False)

        self.sigma = sigma.detach().requires_grad_(False)
        self.ssigma = ssigma.detach().requires_grad_(False)

        self.noise = noiseo.detach().requires_grad_(False)**2
        self.A = V.detach().requires_grad_(False)
        self.hnoise = hnoise.detach()**2
        self.ipoints = ipoints.detach().requires_grad_(False)
        self.sipoints = sipoints.detach().requires_grad_(False)

        return

    # Posterior Inference
    def forward(self, inp, observations, test, testo, ld=1, sig=10, noise=.1, target="squaredexp"):
        # Choose kernel
        kernel = self.squaredExpKernelARD

        ltest = test
        # Get Hyperparms
        if self.trained is False:
            noise, sigma, ld = [1,inp.std().item()**(1/2), (inp.std().item())**(1/2)]
            self.optimizeHyperparms(observations, inp, sigma=sigma, noise=noise, ld=ld ,niter=40,lr=1e-4, kernel=kernel)
            self.trained = True

        posteriorK =[] 
        mu = []
        ltest = ltest.matmul(self.A)
        # Build Null distribution
        for i in range(30):
            if i == 0:
                t = 0
            else:
                t = 1
                p = np.random.permutation(len(ltest.ravel()))
                ltest = ltest.reshape(-1)[p].reshape(*ltest.shape).cuda(self.cuda)
            sigma = (self.sigma+(self.ssigma*np.random.randn()*t)).exp()
            ld = (self.ld +(self.sld*torch.randn(*self.ld.shape).cuda(self.cuda)*t)).exp()
            ipoints = self.ipoints+(self.sipoints*torch.randn(*self.ipoints.shape).cuda(self.cuda)*t)
            tk = sigma*kernel(ltest, ltest, ld)

            # Test given train kernel
            ip = inp.matmul(self.A)
            inptk = sigma*kernel(ltest, ipoints, ld)
            Km = sigma*kernel(ipoints, ipoints, ld) + (self.hnoise).diag()
            Knm = sigma*kernel(ip,ipoints,ld)
            Kmi = torch.inverse(Km) 
            Kn = sigma*kernel(ip,ip,ld)
            lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
            eye = torch.eye(tk.shape[-1]).cuda(self.cuda)
            lni = (1/(lamb+noise*eye).diag()).diag()
            Qm = Km + Knm.matmul(lni).matmul(Knm.T)
            stdMat= torch.inverse(Km) - torch.inverse(Qm)
            muMatr = torch.inverse(Qm).matmul(Knm).matmul(torch.inverse(lamb + self.noise*eye)).matmul(ip)
            # Covariance update
            posteriorK += [(tk - inptk.T.matmul(stdMat).matmul(inptk) + self.noise*eye).slogdet()[1].cpu()]
            posteriormu += inptk.T.matmul(muMat)

        posteriorK = torch.stack(posteriorK)
        posteriormu = torch.stack(posteriormu)

        return posteriormu, posteriorK
