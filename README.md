# gpCCM

Here we extend the convergent cross-mapping method to a Bayesian (approximate) evidence
comparison model given an a priori gaussian processes placed on the observed data. 
We have a model based on variational approximation of the
posterior distribution of the model hyperparameters (we recommend this one) or a
point estimate deterministic hyperparameter model. Effectively, we're placing a
probability distribution on each place in a (reconstructed) state space and calculating the
evidence for a time series Y being caused by X through a conditioned
probability. This reduces to a comparison of a posteriori entropy difference
between H(X|Y) and H(Y|X). If H(X|Y) > H(Y|X), that says that Y provided less
information about X than X did for Y, meaning coupling direction goes from Y to X.

If you use this work, please cite our papers. The first one, for the point
estimate results, can be found in Physical Review E:

```
@article{ghouse2021inferring,
  title={Inferring directionality of coupled dynamical systems using Gaussian process priors: Application on neurovascular systems},
  author={Ghouse, Ameer and Faes, Luca and Valenza, Gaetano},
  journal={Physical Review E},
  volume={104},
  number={6},
  pages={064208},
  year={2021},
  publisher={APS}
}
```

The variational posterior method we submitted to a conference. For the time
being, a link to the article can be found on arxiv:

```
@ARTICLE{ghouse2022parsim,
       author = {{Ghouse}, Ameer and {Valenza}, Gaetano},
        title = "{Inferring Parsimonious Coupling Statistics in Nonlinear Dynamics with Variational Gaussian Processes}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Methodology},
         year = 2022,
        month = mar,
          eid = {arXiv:2203.03868},
        pages = {arXiv:2203.03868},
archivePrefix = {arXiv},
       eprint = {2203.03868},
 primaryClass = {stat.ME},
}
```

Cheers!
