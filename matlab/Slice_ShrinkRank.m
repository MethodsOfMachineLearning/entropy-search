function xx = Slice_ShrinkRank(xx,logP,s0,transpose)
% multivariate slice sampling with shrinking rank covariance adaptation.
% Implemented after Thompson and Neal, November 2010
%
% The algorithm has three inputs:
%   xx   -- the last sample of the Markov Chain
%   logP -- function returning [logP*,dlogP*/dx], the log probability and
%           its first derivative.
%   s0   -- initial proposal width, the one free parameter.
%           Set it to the width of the prior.
%
% Philipp Hennig, August 2011

if transpose; xx = xx'; end

D    = size(xx,1);
[logf,~] = logP(xx');
logy = log(rand()) + logf;

theta= 0.95;

k = 0;
s = s0;
c = zeros(D,0);
J = [];

while true
    k      = k + 1;
    c(:,k) = ProjNullSpace(J,xx + s(k) * randn(D,1));
    sx     = 1 ./ (sum(1./s));
    mx     = sx * sum(bsxfun(@times,1./s,bsxfun(@minus,c,xx)),2);
    xk     = xx + ProjNullSpace(J,mx + sx .* randn(D,1));
    
    [logfk,dlogfk] = logP(xk');
    if logfk > logy     % accept
        if transpose; xx = xk'; else xx = xk; end
        break
    else                % shrink
        g  = ProjNullSpace(J,dlogfk);
        if size(J,2) < D - 1 && g' * dlogfk > 0.5 * norm(g) * norm(dlogfk)
            J      = [J,  g ./ norm(g)];                        %#ok<AGROW>
            s(k+1) = s(k);
        else
            s(k+1) = theta .* s(k); % max(theta .* s(k),s0/1000);
            if s(k+1) < eps
                warning 'contracted down to zero step size. collapsed posterior?'
                if transpose; xx = xx'; return; else return;
                end
            end
        end
    end
end
end

function p = ProjNullSpace(J,v)
    if size(J,2) > 0 
        p = v - J * J' * v;
    else
        p = v;
    end
end
