classdef sum_quadratic
%function [Problem] = sum_quadratic(A, b)
% This file defines sum quadratic problem class
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
% Modified by H.Kasai on Mar. 25, 2018


    properties
        name;    
        dim;
        samples;
        lambda;
        d;
        n;
        A;
        b;
        A_sum;
        b_sum;
        hessain_w_independent;
    end
    
    methods
        function obj = sum_quadratic(A, b) 

            obj.A = A;
            obj.b = b;            

            obj.d = size(obj.b,1);
            obj.n = size(obj.b,2);

            obj.name = 'sum_quadratic';    
            obj.dim = obj.d;
            obj.samples = obj.n;    
            obj.hessain_w_independent = true;

            obj.A_sum = zeros(obj.d,obj.d);
            obj.b_sum = zeros(obj.d,1);
            for j=1:obj.n
                obj.A_sum = obj.A_sum + obj.A(:,:,j);
                obj.b_sum = obj.b_sum + obj.b(:,j);
            end    
        end

        function f = cost(obj, x)

            f = 0;
            for i=1:obj.n
                f = f + 1/2 * x' * obj.A(:,:,i) * x + obj.b(:,i)' * x;
            end
            f = f/obj.n;
        end

        function g = grad(obj, x, indices)

            g = obj.A(:,:,indices) * x + obj.b(:,indices);

        end  

        function g = full_grad(obj, x)

            g = zeros(obj.d,1);
            for i=1:obj.n
                g = g + obj.grad(x,i);
            end
            g = g/obj.n;
        end 

        function h = hess(obj, x, indices)

            h = obj.A(:,:,indices);

        end

        function h = full_hess(obj, x)

            h = zeros(obj.d,obj.d);
            for i=1:obj.n
                h = h + obj.hess(x,i);
            end
            h = h/obj.n;        

        end

        function hv = hess_vec(obj, w, v, indices)

            len = length(indices);

            h = zeros(obj.d,obj.d);
            for i=1:len
                index = indices(i);
                h = h + obj.hess(w,index);
            end
            h = h/len; 

            hv = h*v;

        end

        function w_opt = calc_solution(obj)

            A_inv = zeros(obj.d,obj.d);
            for i=1:d
                A_inv(i,i) = 1/(obj.A_sum(i,i));
            end

            w_opt = -A_inv * obj.b_sum;

        end

        function cn = calc_cn()

            eig_values = eig(obj.A_sum);
            cn = max(eig_values)/min(eig_values);

        end
    end
end

