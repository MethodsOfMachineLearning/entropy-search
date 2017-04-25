function f = Rosenbrock(x)
    a = 1;
    b = 100;
    f = (a-x(:,1)).^2+b.*(x(:,2)-x(:,1).^2).^2;