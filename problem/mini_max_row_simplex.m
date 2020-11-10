classdef mini_max_row_simplex
%function [Problem] = mini_max_row_simplex(A)
%
% Inputs:
%       A           matrix of size dxn.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = max a_i^T x : x in probability simplex
%           where a_1^T, a_2^T, ..., a_n^T are the rows of A
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 29, 2020


    properties
        name;    
        dim;
        samples;
        d;
        n;
        A;
        b;
        prox_flag;
    end
    
    methods
        function obj = mini_max_row_simplex(A) 
            
            obj.A = A;
            
            obj.prox_flag = true;
            
            obj.d = size(obj.A, 1);
            obj.n = size(obj.A, 2);

            obj.name = 'mini_max_row_simplex';    
            obj.dim = obj.d;
            obj.samples = obj.n;

        end
    
        function v = prox(obj, w, t)
            v = proj_simplex(w);
        end    

        function f = cost(obj, w)
            f = max(obj.A * w);
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
        
        % for Subgradient method: A subgradient of f at w is given by a_i(w)
        function subg = full_subgrad(obj, w)
            [~, i] = max(obj.A * w);
            subg = obj.A(i,:)';
        end

        % for Frank-Wolfe, a.k.a. Condtional Gradient
        function [s, idx, sign_flag] = LMO(obj, grad) 
            error('Not implemted yet.'); 
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
            error('Not implemted yet.'); 
        end          

        function w_opt = calc_solution(obj, options_in, method)
            error('Not implemted yet.'); 
        end
    end
end

