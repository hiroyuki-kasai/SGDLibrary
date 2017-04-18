function [Problem] = lasso(A, b, lambda)
% This file defines the lasso (least absolute shrinkage and selection operator) problem for L1 norm. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%       lambda      l1-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/2 * || A * w - b ||^2 + lambda * || w ||_1 ).
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017


    d = size(A, 2);
    n = size(A, 2);
    
    Problem.name = @() 'lasso';    
    Problem.dim = @() d;
    Problem.samples = @() n;
    Problem.lambda = @() lambda;
    
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda);
    end    

    Problem.cost = @cost;
    function f = cost(w)
        reg = reg(w);
        f = 1/2 * sum((A * w - b).^2) + lambda * norm(w,1);
    end

    % calculate l1 norm
    Problem.reg = @reg;
    function r = reg(w)
        r = norm(w,1);
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        g = A' * (A * w - b);
    end

    Problem.grad = @grad;
    function g = grad(w, indices)
        error('Not implemted yet.');
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        error('Not implemted yet.');        
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        error('Not implemted yet.');
    end


end

