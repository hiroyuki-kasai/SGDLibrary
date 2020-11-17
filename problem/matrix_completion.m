classdef matrix_completion
% This file defines a class of a matrix completion problem. 
%
% Inputs:
%       A           Full observation matrix. A.*mask is to be completed. 
%       mask        Linear operator to extract existing elements, denoted as P_omega( ).
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min     f(w) =  1/2 ||P_omega(W) - P_omega(A)||_2^2,
%           s.t.    || W ||_* <= 1,
%
% "w" is the model parameter of size mxn vector, which is transformed to matrix W of size [m, n]. 
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 16, 2020


    properties
        name;    
        dim;
        samples;
        m;
        n;
        A;
        mask;
        r;
        prox_flag;
    end
    
    methods
        function obj = matrix_completion(A, mask, r, varargin)

            obj.A = A;
            obj.mask = mask;
            obj.r = r;
            
            obj.m = size(obj.A, 1);
            obj.n = size(obj.A, 2);
            
            obj.name = 'matrix completion';  
            obj.dim = obj.m * obj.n;
            obj.samples = obj.n;
            
            obj.prox_flag = true;
        end
        
        function v = prox(obj, w, t)
            w_mat = reshape(w, [obj.m obj.n]);
            v_mat = proj_nuclear_ball(w_mat, obj.r);
            v = v_mat(:);
        end
        
        function [uvt, idx, sign_flag] = LMO(obj, grad)
            
            grad_mat = reshape(grad, [obj.m obj.n]);

            [u,s,v] = svds(-grad_mat, 1); % LMO is solved by top singular vector pair
            uvt = obj.r * u*v';
            uvt = uvt(:);
            idx = 1;
            sign_flag = 1;

        end         

        function f = cost(obj, w)
            L = reshape(w, [obj.m obj.n]);
            diff = (L - obj.A.*obj.mask) .* (obj.A.*obj.mask ~= 0);

            f = 1/2 * norm(diff, 'fro')^2;
        end
        
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
        
        
        function w_opt = calc_solution(obj, method, options_in)

            if nargin < 2
                method = 'ag';
            end        

            options.max_epoch = options_in.max_iter;
            options.w_init = options_in.w_init;
            options.verbose = options_in.verbose;
            options.tol_optgap = 1.0e-24;
            options.tol_gnorm = 1.0e-16;
            options.step_alg = 'backtracking';
            
            if ~options.verbose
                fprintf('Calclation of solution started ... ');
            end            

            if strcmp(method, 'sd')
                [w_opt,~] = sd(obj, options);
            elseif strcmp(method, 'cg')
                [w_opt,~] = ncg(obj, options);
            elseif strcmp(method, 'newton')
                options.sub_mode = 'INEXACT';    
                options.step_alg = 'non-backtracking'; 
                [w_opt,~] = newton(obj, options);
            elseif strcmp(method, 'ag')
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_opt,~] = ag(obj, options);            
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end
            
            if ~options.verbose
                fprintf('done\n');
            end             
        end          
        
    end


end

