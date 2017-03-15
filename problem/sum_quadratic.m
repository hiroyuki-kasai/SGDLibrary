function [Problem] = sum_quadratic(A, b)
% This file defines sum quadratic problem
%
% Inputs:
%           A(d:d:n)    n matrix of size dxd for n samples
%           b(d:n)      n column vectors of size d for n samples
%
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * (sum_i^n 1/2 * w^T * A_i * w + b_i^T * w).
%           where 
%           w in R^d
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Mar. 13, 2017


    d = size(b,1);
    n = size(b,2);
    
    Problem.name = @() 'sum_quadratic';    
    Problem.dim = @() d;
    Problem.samples = @() n;    
    Problem.A = @() A;     
    Problem.b = @() b; 
    Problem.hessain_w_independent = @() true;
    
    A_sum = zeros(d,d);
    b_sum = zeros(d,1);
    for j=1:n
        A_sum = A_sum + A(:,:,j);
        b_sum = b_sum + b(:,j);
    end    

    Problem.cost = @cost;
    function f = cost(x)

        f = 0;
        for i=1:n
            f = f + 1/2 * x' * A(:,:,i) * x + b(:,i)' * x;
        end
        f = f/n;
    end

    Problem.grad = @grad;
    function g = grad(x, indices)
        
        g = A(:,:,indices) * x + b(:,indices);
        
    end  

    Problem.full_grad = @full_grad;
    function g = full_grad(x)
        
        g = zeros(d,1);
        for i=1:n
            g = g + grad(x,i);
        end
        g = g/n;
    end 

    Problem.hess = @hess; 
    function h = hess(x, indices)
        
        h = A(:,:,indices);
        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(x)
        
        h = zeros(d,d);
        for i=1:n
            h = h + hess(x,i);
        end
        h = h/n;        
        
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        
        len = length(indices);
        
        h = zeros(d,d);
        for i=1:len
            index = indices(i);
            h = h + hess(w,index);
        end
        h = h/len; 
        
        hv = h*v;
        
    end

    Problem.calc_solution = @calc_solution; 
    function w_opt = calc_solution()
        
        A_inv = zeros(d,d);
        for i=1:d
            A_inv(i,i) = 1/(A_sum(i,i));
        end
        
        w_opt = -A_inv * b_sum;
        
    end

    Problem.calc_cn = @calc_cn; 
    function cn = calc_cn()
        
        eig_values = eig(A_sum);
        cn = max(eig_values)/min(eig_values);
        
    end
end

