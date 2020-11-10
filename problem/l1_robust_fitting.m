classdef l1_robust_fitting
%function [Problem] = l1_robust_fitting(A, b, lambda)
% This file defines the l1_robust_fitting problem with L1 norm. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%       lambda_c    l1-norm composite term parameter. 
%       lambda_r    l1-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = || A * w - b ||_1 + lambda_c * || w ||_1.
%
%           or
%
%           min f(w) = || A * w - b ||_1 + lambda_r * || w ||_1.
%
% "w" is the model parameter of size d vector.
%
% Note that the former can be solved by subgradient descent while 
% the latter can be doen by proximal subgradient descent.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 21, 2020


    properties
        name;    
        dim;
        samples;
        lambda_c;
        lambda_r;
        d;
        n;
        A;
        b;
        prox_flag;
    end
    
    methods
        function obj = l1_robust_fitting(A, b, lambda_c, lambda_r) 
            
            obj.A = A;
            obj.b = b;
            
            if (lambda_c > 0) && (lambda_r > 0)
                error('Not correctly specified for lambdas.');
            elseif lambda_r > 0
                obj.prox_flag = true;
            else
                obj.prox_flag = false;
            end
            
            obj.lambda_c = lambda_c;
            obj.lambda_r = lambda_r;            
            
            obj.d = size(obj.A, 2);
            obj.n = size(obj.A, 2);

            obj.name = 'l1_robust_fitting';    
            obj.dim = obj.d;
            obj.samples = obj.n;

        end
    
        function v = prox(obj, w, t)
            if obj.prox_flag
                v = soft_thresh(w, t * obj.lambda_r);
            else
                % Just copy
                v = w;
            end
        end    

        function f = cost(obj, w)
            reg = obj.reg(w);
            f = sum(abs(obj.A * w - obj.b)) + (obj.lambda_c + obj.lambda_r) * reg;
        end

        % calculate l1 norm
        function r = reg(obj, w)
            r = norm(w,1);
        end

        function r = residual(obj, w)
            error('Not implemted yet.'); 
        end

        function f = cost_batch(obj, w, indices)
            error('Not implemted yet.');        
        end

        function g = full_grad(obj, w)
            g = zeros(size(w));
        end

        function g = grad(obj, w, indices)
            g = zeros(size(w));       
        end
        
        function subg = full_subgrad(obj, w)
            subg = (obj.A)' * sign(obj.A * w - obj.b) + obj.lambda_c * sign(w);
        end        

        function h = hess(obj, w, indices)
            error('Not implemted yet.');        
        end

        function h = full_hess(obj, w)
            error('Not implemted yet.');      
        end

        function hv = hess_vec(obj, w, v, indices)
            error('Not implemted yet.');
        end

        function w_opt = calc_solution(obj, options_in, method)
            error('Not implemted yet.');
        end
    end
end

