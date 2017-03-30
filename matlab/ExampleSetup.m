% setup for Entropy Search with a demo function
%
% (C) 2011,2017 Max-Planck-Institute for Intelligent Systems
%
% Authors: Philipp Hennig & Christian Schuler, 2011
%          Edgar D. Klenske, 2017

%% set up the dependencies

% in these two lines, you have to add the paths to 
run ext/gpml/startup.m  % gpml toolbox
addpath ext/logsumexp/  % logsumexp package
addpath ext/tprod/      % tprod package

%% set up and run entropy search

% set up prior belief
N               = 2; % number of input dimensions
in.covfunc      = {@covSEard};       % GP kernel
in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.
hyp.cov         = log([ones(N,1);1]); % hyperparameters for the kernel
hyp.lik         = log([1e-3]); % noise level on signals (log(standard deviation));
in.hyp          = hyp;  % hyperparameters, with fields .lik (noise level) and .cov (kernel hyperparameters), see documentation of the kernel functions for details.

% should the hyperparameters be learned, too?
in.LearnHypers  = true; % yes.
in.HyperPrior   = @SEGammaHyperPosterior;

% constraints defining search space:
in.xmin         = [-1,-1]; % lower bounds of rectangular search domain
in.xmax         = [2,2]; % upper bounds of rectangular search domain
in.MaxEval      = 60;    % Horizon (number of evaluations allowed)

% objective function:
in.f            = @(x) Rosenbrock(x); % handle to objective function

result = EntropySearch(in); % the output is a struct which contains GP datasets, which can be evaluated with the gpml toolbox.

%% visualize the result

% plot the resulting surrogate function

GP = result.GP; % store for easier use

% define a grid for the function evaluation
r = 25;
x = linspace(in.xmin(1), in.xmax(1), r);
y = linspace(in.xmin(2), in.xmax(2), r);
[X, Y] = meshgrid(x, y);
V = [X(:), Y(:)];

% evaluate the GP on a grid
[Zm, Zs] = gp(GP.hyp, @infExact, [], GP.covfunc, GP.likfunc, GP.x, GP.y, V);
Zm = reshape(Zm, size(X));
Zs = reshape(sqrt(Zs), size(X));

% plot the resulting surface
clf; hold on; view(3)
surf(X,Y,Zm,'FaceAlpha', 0.8, 'EdgeAlpha', 0.4)
surf(X,Y,Zm + 2*Zs,'FaceAlpha', 0.3, 'EdgeAlpha', 0.15)
surf(X,Y,Zm - 2*Zs,'FaceAlpha', 0.3, 'EdgeAlpha', 0.15)
plot3(GP.x(:,1), GP.x(:,2), GP.y, 'or')


