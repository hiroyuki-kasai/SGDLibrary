classdef lasso
%function [Problem] = lasso(A, b, lambda)
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
%           min f(w) = 1/2 * || A * w - b ||_2^2 + lambda * || w ||_1
%
%           or
%
%           min f(w) = 1/2 * || A * w - b ||_2^2,      s.t. || w ||_1 < r.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017
% Modified by H.Kasai on Mar. 25, 2018

    properties
        name;    
        dim;
        samples;
        lambda;
        r;
        d;
        n;
        A;
        b;
        AtA;
        Atb;
        L;
        prox_flag;
        sign_flag;
        idx;
    end
    
    methods
        function obj = lasso(A, b, lambda, r) 
            
            obj.A = A;
            obj.b = b;
            
            if (lambda > 0) && (r > 0)
                error('Not correctly specified for lambda or r.');
            elseif lambda > 0
                obj.prox_flag = true;
            else % r > 0
                obj.prox_flag = false;                
            end
            
            obj.lambda = lambda;
            obj.r = r;

            obj.d = size(obj.A, 2);
            obj.n = size(obj.A, 2);

            obj.name = 'lasso';    
            obj.dim = obj.d;
            obj.samples = obj.n;

            obj.AtA = obj.A'*obj.A;
            obj.Atb = obj.A'*obj.b;
            %L = max(eig(AtA));
            fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
            obj.L = eigs(obj.AtA, 1);
            fprintf('is L=%f.\n', obj.L);
        end
    
        function v = prox(obj, w, t)
            if obj.prox_flag
                v = soft_thresh(w, t * obj.lambda);
            else
                v = sign(w) .* proj_simplex(abs(w), obj.r, 'ineq');
            end
        end    

        function f = cost(obj, w)
            reg = obj.reg(w);
            f = 1/2 * sum((obj.A * w - obj.b).^2) + obj.lambda * reg;
        end

        % calculate l1 norm
        function r = reg(obj, w)
            r = norm(w,1);
        end

        function r = residual(obj, w)
            r = - obj.A * w + obj.b;
        end

        function f = cost_batch(obj, w, indices)
            error('Not implemted yet.');        
        end

        function g = full_grad(obj, w)
            %g = obj.A' * (obj.A * w - obj.obj.b);
            g = obj.AtA * w - obj.Atb;
        end

        function g = grad(obj, w, indices)
            A_partial = obj.A(:,indices);
            g = A_partial' * (A_partial * w - obj.b);        
        end
        
        % for Subgradient descebt
        function subg = full_subgrad(obj, w)
            subg = obj.lambda * sign(w);
        end

        % for Frank-Wolfe, a.k.a. Condtional Gradient
        function [s, idx, sign_flag] = LMO(obj, grad) 
            [val, idx] = max( abs(grad) );
            obj.sign_flag = -1 * sign(grad(idx));
            obj.idx = idx;

            s = zeros(size(grad)); 
            s(idx) = obj.sign_flag * obj.r; 
            
            sign_flag = obj.sign_flag;
        end           

        function h = hess(obj, w, indices)
            error('Not implemted yet.');        
        end

        function h = full_hess(obj, w)
            h = obj.AtA;       
        end

        function hv = hess_vec(obj, w, v, indices)
            error('Not implemted yet.');
        end
        
        function step = exact_line_search(obj, w, dir, idx, sign_flag)
            As = sign_flag * obj.r * obj.A(:,idx); % = the i-th column of the dictionary matrix A
            As_minus_Ax = As - obj.A*w;
            step = max(0, min(1, As_minus_Ax' * (obj.b-obj.A*w) / (As_minus_Ax' * As_minus_Ax)));
        end          

        function w_opt = calc_solution(obj, options_in, method)

            if nargin < 3
                method = 'ag';
            end        

            options.max_iter = options_in.max_iter;
            options.w_init = options_in.w_init;
            options.verbose = true;
            options.tol_optgap = 1.0e-24;
            options.tol_gnorm = 1.0e-16;
            options.step_alg = 'backtracking';

            if strcmp(method, 'sd')
                [w_opt,~] = sd(obj, options);
            elseif strcmp(method, 'cg')
                [w_opt,~] = ncg(obj, options);
            elseif strcmp(method, 'newton')
                options.sub_mode = 'INEXACT';    
                options.step_alg = 'non-backtracking'; 
                [w_opt,~] = newton(obj, options);
            elseif strcmp(method, 'ag')
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_opt,~] = ag(obj, options);            
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end
        end
    end
end

