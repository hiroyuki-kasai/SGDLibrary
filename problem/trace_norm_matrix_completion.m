classdef trace_norm_matrix_completion
% This file defines a class of a matrix completion problem with trace (nuclear) norm minimization. 
%
% Inputs:
%       A           Full observation matrix. A.*mask is to be completed. 
%       mask        Linear operator to extract existing elements, denoted as P_omega( ).
%       lambda      l1-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) =  1/2 ||P_omega(W) - P_omega(A)||_2^2 + lambda * ||X||_*,
%
% "w" is the model parameter of size mxn vector, which is transformed to matrix W of size [m, n]. 
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
        m;
        n;
        A;
        mask;
        prox_flag;
    end
    
    methods
        function obj = trace_norm_matrix_completion(A, mask, varargin)

            obj.A = A;
            obj.mask = mask;
            
            if nargin < 3
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end   
            
            if obj.lambda > 0
                obj.prox_flag = true;
            else
                obj.prox_flag = false;
            end
            
            obj.m = size(obj.A, 1);
            obj.n = size(obj.A, 2);

            obj.name = 'matrix completion';  
            obj.dim = obj.m * obj.n;
            obj.samples = obj.n;
        end

        function v = prox(obj, w, t)
            v = svd_shrink(w, t * obj.lambda, [obj.m obj.n]);
        end    

        function f = cost(obj, w)
            L = reshape(w, [obj.m obj.n]);
            diff = (L - obj.A.*obj.mask) .* (obj.A.*obj.mask ~= 0);
            trace_norm = reg(obj, w);

            f = 1/2 * norm(diff, 'fro')^2 + obj.lambda * trace_norm;
        end

        % calculate trace norm
        function r = reg(obj, w)
            L = reshape(w, [obj.m obj.n]);
            [~,S,~] = svd(L,'econ');
            s = diag(S);
            r = sum(s);
        end

        function f = cost_batch(obj, w, indices)
            error('Not implemted yet.');        
        end

        function g = full_grad(obj, w)
            L = reshape(w, [obj.m obj.n]);
            G = (L - obj.A.*obj.mask) .* (obj.A.*obj.mask ~= 0);
            g = G(:);
        end

        function g = grad(obj, w, indices)
            error('Not implemted yet.');
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
        
    end


end

