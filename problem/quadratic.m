function [Problem] = quadratic(A, b)
% This file defines quadratic problem
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
% Modified by H.Kasai on Oct. 31, 2016


    d = length(b);
    n = d;
    
    Problem.name = @() 'quadratic';    
    Problem.dim = @() d;
    Problem.samples = @() n;    
    Problem.A = @() A;     
    Problem.b = @() b;  
    Problem.hessain_w_independent = @() true;

    Problem.cost = @cost;
    function f = cost(x)

        f = 1/2 * x' * A * x - b' * x;
    end

    Problem.grad = @grad;
    function g = grad(x)
        
        g = A * x - b;
        
    end  

    Problem.full_grad = @full_grad;
    function g = full_grad(x)
        
        g = grad(x);
        
    end 

    Problem.hess = @hess; 
    function h = hess(x)
        
        h = A;
        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(x)
        
        h = hess(x);
        
    end

end

