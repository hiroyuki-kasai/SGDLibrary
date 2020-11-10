classdef l1_robust_fitting_simplex
%function [Problem] = l1_robust_fitting_simplex(A, b)
% This file defines the l1_robust_fitting_simplex problem with probablity simplex constraint. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = || A * w - b ||_1,   s.t. w in probablity simplex.
%
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
        proj_type
    end
    
    methods
        function obj = l1_robust_fitting_simplex(A, b, varargin) 
            
            obj.A = A;
            obj.b = b;
            obj.prox_flag = true;
            
            if nargin < 3
                obj.proj_type = 'euclidean';
            else
                obj.proj_type = varargin{1};
            end            


            obj.d = size(obj.A, 2);
            obj.n = size(obj.A, 2);

            obj.name = 'l1_robust_fitting_simplex';    
            obj.dim = obj.d;
            obj.samples = obj.n;

        end
    
        function v = prox(obj, w, t)
            if strcmp(obj.proj_type, 'euclidean')
                v = proj_simplex(w, 1, 'eq');
            elseif strcmp(obj.proj_type, 'bregman')
                v = w / norm(w, 1);
            else
                error('Not implemted yet.'); 
            end
        end    

        function f = cost(obj, w)
            f = sum(abs(obj.A * w - obj.b));
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
            subg = (obj.A)' * sign(obj.A * w - obj.b);
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
        
        % Update for Mirror Descent
        function w_out = md_update(obj, w, step)
            w_out = w .* exp(-step * obj.full_subgrad(w)); 
            %w_out = w_out / sum(w_out); This can be done by bregman projection
        end
    end
end

