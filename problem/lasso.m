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
%           min f(w) = 1/2 * || A * w - b ||_2^2 + lambda * || w ||_1 ).
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017
% Modified by H.Kasai on Sep. 25, 2017


    d = size(A, 2);
    n = size(A, 2);
    
    Problem.name = @() 'lasso';    
    Problem.dim = @() d;
    Problem.samples = @() n;
    Problem.lambda = @() lambda;
    Problem.A = @() A;    
    Problem.b = @() b; 
    
    AtA = A'*A;
    Problem.AtA = @() AtA;
    Atb = A'*b;
    %L = max(eig(AtA));
    fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
    L = eigs(A'*A,1);
    fprintf('is L=%f.\n', L);
    Problem.L = @() L;
    
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda);
    end    

    Problem.cost = @cost;
    function f = cost(w)
        reg = reg(w);
        f = 1/2 * sum((A * w - b).^2) + lambda * reg;
    end

    % calculate l1 norm
    Problem.reg = @reg;
    function r = reg(w)
        r = norm(w,1);
    end

    Problem.residual = @residual;
    function r = residual(w)
        r = - A * w + b;
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        %g = A' * (A * w - b);
        g = AtA * w - Atb;
    end

    Problem.grad = @grad;
    function g = grad(w, indices)
        A_partial = A(:,indices);
        g = A_partial' * (A_partial * w - b);        
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
        error('Not implemted yet.');        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        h = AtA;       
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        error('Not implemted yet.');
    end

    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, method, options_in)
        
        if nargin < 2
            method = 'gd_nesterov';
        end        
        
        options.max_iter = options_in.max_iter;
        options.w_init = options_in.w_init;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;
        options.tol_gnorm = 1.0e-16;
        options.step_alg = 'backtracking';
        
        if strcmp(method, 'sg')
            [w_opt,~] = gd(problem, options);
        elseif strcmp(method, 'cg')
            [w_opt,~] = ncg(problem, options);
        elseif strcmp(method, 'newton')
            options.sub_mode = 'INEXACT';    
            options.step_alg = 'non-backtracking'; 
            [w_opt,~] = newton(problem, options);
        elseif strcmp(method, 'gd_nesterov')
            options.step_alg = 'backtracking';
            options.step_init_alg = 'bb_init';
            [w_opt,~] = gd_nesterov(problem, options);            
        else 
            options.step_alg = 'backtracking';  
            options.mem_size = 5;
            [w_opt,~] = lbfgs(problem, options);              
        end
    end
end

