function [y] = logsumexp(X)
% Calculate log sum of exponentials, i.e., log sum_{l=1}^L exp(X)
%
% Input
%       X     LxN matrix, i.e., x_{l,n}
% Output
%       y     log sum of exponentials, i.e., log sum_{l=1}^L exp(X)
%
%
%   Here, we define
%   max_X = {max_1,...., max_n,...,max_N }, where max_n = argmax_{l}
%   x_{l,n}.
%
%   Addressing n-th commun of X, i.e., X(:,n)=x_{l,n}_{l=1}^L, 
%   we obtain
%   log sum_{l=1}^L exp(x_{l,n}) 
%           = log (sum_{l=1}^L exp(max_n)/(exp(max_n) exp(x_{l,n}))
%           = log (exp(max_n) sum_{l=1}^L 1/(exp(max_n) exp(x_{l,n}))
%           = log (exp(max_n) + log (sum_{l=1}^L (exp(-max_n) exp(x_{l,n}))
%           = max_n + log (sum_{l=1}^L exp(x_{l,n}-max_n)).
%
%    
% This file is part of SGDLibrary.
%
% Created H.Kasai on Oct. 19, 2016

    L = size(X, 1);
    max_X = max(X);
    Diff_X = X - ones(L, 1) * max_X;
    y = max_X + log(sum(exp(Diff_X))); 
end

    
    
