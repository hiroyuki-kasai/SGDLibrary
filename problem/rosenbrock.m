function [Problem] = rosenbrock(d)
% This file defines Rosenbrock problem
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
% Modified by H.Kasai on Oct. 31, 2016


    n = d;

    Problem.name = @() 'rosenbrock';    
    Problem.dim = @() d;
    Problem.samples = @() n; 
    Problem.hessain_w_independent = @() false;

    Problem.cost = @cost;
    function f = cost(x)

        f = sum(100*(x(2:d)-x(1:d-1).^2).^2 + (1-x(1:d-1)).^2);
    end

    Problem.grad = @grad;
    function g = grad(x)
        
        g = zeros(d, 1);
        g(1:d-1) = - 400*x(1:d-1).*(x(2:d)-x(1:d-1).^2) - 2*(1-x(1:d-1));
        g(2:d) = g(2:d) + 200*(x(2:d)-x(1:d-1).^2);
        
    end  

    Problem.full_grad = @full_grad;
    function g = full_grad(x)
        
        g = grad(x);
        
    end 

    Problem.hess = @hess; 
    function h = hess(x)
        
        h = zeros(d,d);
        h(1:d-1,1:d-1) = diag(-400*x(2:d) + 1200*x(1:d-1).^2 + 2);
        h(2:d,2:d) = h(2:d,2:d) + 200*eye(d-1);
        h = h - diag(400*x(1:d-1),1) - diag(400*x(1:d-1),-1);
        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(x)
        
        h = hess(x);
        
    end

    Problem.calc_solution = @calc_solution; 
    function w_opt = calc_solution()
        
        w_opt = ones(d,1);
        
    end


end

