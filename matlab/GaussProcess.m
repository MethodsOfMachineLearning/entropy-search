function f = GaussProcess(x)
    rng(2017) % fix random number generator to get consistent function values
    
    l = 1;
    covSE1 = @(a,b) exp(-0.5*bsxfun(@minus,a(1,:),b(1,:)').^2./l^2);
    covSE2 = @(a,b) exp(-0.5*bsxfun(@minus,a(2,:),b(2,:)').^2./l^2);

    cov = @(a,b) covSE1(a,b) .* covSE2(a,b);
    
    % define a grid for the function evaluation
    r = 25;
    xl = linspace(-2, 2, r);
    yl = linspace(-2, 2, r);
    [X, Y] = meshgrid(xl, yl);
    V = [X(:), Y(:)];

    % evaluate the GP on a grid
    Zm = chol(cov(V',V')+1e-3*eye(size(V,1)))'*randn(size(V,1),1);
    Zm = reshape(Zm, size(X));

    % function evaluation by interpolation
    f = interp2(X,Y,Zm,x(:,1),x(:,2));