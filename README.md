Entropy Search for Information-Efficient Global Optimization -- Demo Code
=========================================================================

VERSION 1.1 -- released March 2017

This is Matlab demonstration code for Entropy Search, as described in http://www.jmlr.org/papers/v13/hennig12a.html

Dependencies
------------

* the Matlab optimization toolbox. If you do not have this, you can go through the code and replace calls to fmincon with fminbnd (much less efficient), or with a call to minimize.m (which you can get from http://www.gaussianprocess.org/gpml/code/matlab/util/minimize.m). But note that minimize does not automatically handle linear constraints. You can implement those naively by changing function handles such that they return +inf whenever evaluated outside the bounds.

All other dependencies (Eigen, logsumexp, tprod) are currently part of this repository.

Running Entropy Search
----------------------

From the matlab subdirectory, you should be able to call

EntropySearch(in), where

in.covfunc      = {@covSEard};       % GP kernel  
in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.  
in.hyp          = hyp;  % hyperparameters, with fields .lik (noise level) and .cov (kernel hyperparameters), see documentation of the kernel functions for details.  
in.xmin         = xmin; % lower bounds of rectangular search domain  
in.xmax         = xmax; % upper bounds of rectangular search domain  
in.MaxEval      = H;    % Horizon (number of evaluations allowed)  
in.f            = @(x) f(x) % handle to objective function  

That handle @f is obviously the core part of the problem. If you use this method for actual experimental design, use the "PhysicalExperiment" function handle, which simply prompts for user input at the selected locations.


Demo Experiment
---------------

An example script can be found in `ExampleSetup.m`, it tries to find an optimum for the Rosenbrock function and plots the resulting Gaussian process afterwards.


Compiling
---------

Part by part, the code of Entropy Search will be replaced by cpp/mex implementations to achieve a considerable speedup. The Matlab version of the replaced code resides in the `util` subdirectory, while the c++ implementation is in `cpp`. To compile, run the script `compile_utilities.m`. The mex-files take precendence over the Matlab implementation.


Copyright
---------

(C) 2011, 2017 Max Planck Institute for Intelligent Systems

Philipp Hennig & Christian Schuler, 2011  
Edgar D. Klenske, 2017
