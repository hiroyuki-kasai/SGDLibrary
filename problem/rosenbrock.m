classdef rosenbrock
% This file defines Rosenbrock problem class
%
% Inputs:
%           d       dimension of paramter
%
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%       min f(x) = sum_{i=1:d-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2 
%
% where d is the dimension of x. The true minimum is 0 at x = (1 1 ... 1).
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 29, 2016
% Modified by H.Kasai on Mar. 25, 2018


    properties
        name;    
        dim;
        samples;
        d;
        n;
        hessain_w_independent;
    end
    
    methods
        function obj = rosenbrock(d) 
            
            obj.d = d;
            obj.n = obj.d;

            obj.name = 'rosenbrock';    
            obj.dim = obj.d;
            obj.samples = obj.n; 
            obj.hessain_w_independent = false;
        end

        function f = cost(obj, x)

            f = sum(100*(x(2:obj.d)-x(1:obj.d-1).^2).^2 + (1-x(1:obj.d-1)).^2);
        end

        function g = grad(obj, x)

            g = zeros(obj.d, 1);
            g(1:obj.d-1) = - 400*x(1:obj.d-1).*(x(2:obj.d)-x(1:obj.d-1).^2) - 2*(1-x(1:obj.d-1));
            g(2:obj.d) = g(2:obj.d) + 200*(x(2:obj.d)-x(1:obj.d-1).^2);

        end  

        function g = full_grad(obj, x)

            g = obj.grad(x);

        end 

        function h = hess(obj, x)

            h = zeros(obj.d,obj.d);
            h(1:obj.d-1,1:obj.d-1) = diag(-400*x(2:obj.d) + 1200*x(1:obj.d-1).^2 + 2);
            h(2:obj.d,2:obj.d) = h(2:obj.d,2:obj.d) + 200*eye(obj.d-1);
            h = h - diag(400*x(1:obj.d-1),1) - diag(400*x(1:obj.d-1),-1);

        end

        function h = full_hess(obj, x)

            h = obj.hess(x);

        end

        function w_opt = calc_solution(obj)

            w_opt = ones(obj.d,1);

        end
    end
end

