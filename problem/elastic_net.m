function [Problem] = elastic_net(A, b, lambda1, lambda2)
% This file defines the Elastic Net problem. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%       lambda1     l1-regularized parameter. 
%       lambda2     l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + 1/2 * lambda2 * || w ||_2^2 + lambda1 * || w ||_1 ).
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017


    d = size(A, 2);
    n = size(A, 2);
    
    Problem.name = @() 'elastic net';    
    Problem.dim = @() d;
    Problem.samples = @() n;
    Problem.lambda = @() lambda;
    Problem.A = @() A;    
    Problem.b = @() b; 
    
    AtA_l2 = A'*A + lambda2*eye(n);
    Problem.AtA_l2 = @() AtA_l2;
    
    Atb = A'*b;
     
    
    fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
    L = eigs(A'*A,1);
    fprintf('is L=%f.\n', L);
    Problem.L = @() L;
    
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda1);
    end    

    Problem.cost = @cost;
    function f = cost(w)
        reg = reg(w);
        f = 1/2 * sum((A * w - b).^2) + 1/2 * lambda2 * norm(w,2)^2 + lambda1 * reg;
    end

    % calculate l1 norm
    Problem.reg = @reg;
    function r = reg(w)
        r = norm(w,1);
    end

    Problem.residual = @residual;
    function r = residual(w, i)
        %r = - A * w + b;
        %r = - AtA_l2 * w + Atb; 
        r = - (A(:, i)' * A + lambda2) * w + A(:, i)'*b; 
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        %g = A' * (A * w + lambda2 - b);
        g = AtA_l2 * w - Atb;
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
        h = AtA_l2;       
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        error('Not implemted yet.');
    end


end

