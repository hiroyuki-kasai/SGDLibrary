classdef quadratic
% This file defines quadratic problem class
%
% Inputs:
%           A       a positive definite matrix of size dxd
%           b       a column vector of size d
%
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(x) = 1/2 * x^T * A * x - b^T * x.
%           where 
%           x in R^d
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 29, 2016
% Modified by H.Kasai on Mar. 25, 2018

    properties
        d;
        n;  
        name;
        dim;
        samples;
        A;
        b;
        hessain_w_independent;
        L;
        mu;
        cn;
        prox_flag;
    end
    
    methods
        function obj = quadratic(A, b)
            obj.d = length(b);
            obj.n = obj.d;
            obj.prox_flag = false;

            obj.name = 'quadratic';    
            obj.dim = obj.d;
            obj.samples = obj.n;    
            obj.A = A;     
            obj.b = b;  
            obj.hessain_w_independent = true;
            eigvalues = eigs(obj.A);
            obj.L = max(eigvalues);
            obj.mu = min(eigvalues);
            obj.cn = obj.L/obj.mu; % = cond(onj.A)
            fprintf('L = %f, mu = %f, cn = %f\n', obj.L, obj.mu, obj.cn);
        end

        function f = cost(obj, x)

            f = 1/2 * x' * obj.A * x - obj.b' * x;
        end

        function g = grad(obj, x)

            g = obj.A * x - obj.b;

        end  

        function g = full_grad(obj, x)

            g = obj.grad(x);

        end 

        function h = hess(obj, x)

            h = obj.A;

        end

        function h = full_hess(obj, x)

            h = obj.hess(x);

        end
    end

end

