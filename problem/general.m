function [Problem] = general(f_in, g_in, h_in, hv_in, d)
% This file defines quadratic problem
%
% Inputs:
%       f           cost function
%       g           gradient
%       h           hessian
%       hv          hessian-product 
%       d           dimension
%
% Output:
%       Problem     problem instance. 
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 29, 2016


    Problem.name = @() 'general';  
    Problem.dim = @() d;
    Problem.samples = @() 0; 
    Problem.hessain_w_independent = @() false;

    Problem.cost = @cost;
    function f = cost(x)

        f = f_in(x);
    end

    Problem.grad = @grad;
    function g = grad(x, indices)
        
        g = g_in(x);
        
    end        

    Problem.full_grad = @full_grad;
    function g = full_grad(x)
        
        g = g_in(x);
        
    end

    Problem.hess = @hess; 
    function h = hess(x)
        
        h = h_in(x);
        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(x)
        
        h = hess(x);
        
    end

    Problem.hess_vec = @hess_vec; 
    function h = hess_vec(x)
        
        h = hv_in(x);
        
    end

end

