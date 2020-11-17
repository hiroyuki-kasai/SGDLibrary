classdef lasso
%function [Problem] = lasso(A, b, lambda, sub_mode, smooth_lambda)
% This file defines the lasso (least absolute shrinkage and selection operator) problem for L1 norm. 
%
% Inputs:
%       A               dictionary matrix of size dxn.
%       b               observation vector of size dx1.
%       lambda          l1-regularized parameter. 
%       sub_mode        {'prox_reg', 'subgrad_reg', 'smooth', 'const'}
%       smooth_lambda
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%       (1) sub_mode = 'prox_reg' for L1 regularizer solved by proximal methods
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + lambda * || w ||_1
%
%       (2) sub_mode = 'subgrad_reg' for L1 regularizer solved by subgrad methods
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + lambda * || w ||_1
%
%       (3) sub_mode = 'smooth' for smoothing 
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + smooth_lambda * || w ||_1.
%
%       (43) sub_mode = 'const' for L1 constraint 
%
%           min f(w) = 1/2 * || A * w - b ||_2^2,      s.t. || w ||_1 < r
%
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017
% Modified by H.Kasai on Mar. 25, 2018
% Modified by H.Kasai on Mar. 27, 2020
% Modified by H.Kasai on Mar. 30, 2020


    properties
        name;    
        dim;
        samples;
        sub_mode;
        lambda;
        d;
        n;
        A;
        b;
        AtA;
        Atb;
        L;
        prox_flag;
        sign_flag;
        idx;
        smooth_mu;      % for smoothing
        smooth_type;    % for smoothing
    end
    
    methods
        function obj = lasso(A, b, lambda, sub_mode, varargin) 
            
            obj.A = A;
            obj.b = b;
            
            if strcmp(sub_mode, 'prox_reg')
                obj.prox_flag = true;
            elseif strcmp(sub_mode, 'subgrad_reg')
                obj.prox_flag = false;                
            elseif strcmp(sub_mode, 'const')
                obj.prox_flag = true; 
            elseif strcmp(sub_mode, 'smooth')
                obj.prox_flag = false;
            else
                error('Not correctly specified for sub_mode');
            end

            obj.sub_mode = sub_mode;
            obj.lambda = lambda;
            
            if nargin < 6
                obj.smooth_type = 'type1';
            else
                obj.smooth_type = varargin{2};
            end             
            
            if nargin < 5
                obj.smooth_mu = 0.1;
            else
                obj.smooth_mu = varargin{1};
            end            

            obj.d = size(obj.A, 2);
            obj.n = size(obj.A, 2);

            obj.name = 'lasso';    
            obj.dim = obj.d;
            obj.samples = obj.n;

            obj.AtA = obj.A'*obj.A;
            obj.Atb = obj.A'*obj.b;
            %L = max(eig(AtA));
            fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
            obj.L = eigs(obj.AtA, 1);
            fprintf('is L = %f.\n', obj.L);
        end
    
        function v = prox(obj, w, t)
            if strcmp(obj.sub_mode, 'prox_reg')
                v = soft_thresh(w, t * obj.lambda);
            elseif strcmp(obj.sub_mode, 'const')
                %v = sign(w) .* proj_simplex(abs(w), obj.lambda, 'ineq');
                v = proj_l1_ball(w, obj.lambda);
            else
                error('Not correctly specified for sub_mode');
            end
        end    

        function f = cost(obj, w)
            reg = obj.reg(w);
            f = 1/2 * sum((obj.A * w - obj.b).^2) + obj.lambda * reg;
        end

        % calculate l1 norm
        function r = reg(obj, w)
            r = norm(w,1);
        end

        function r = residual(obj, w)
            r = - obj.A * w + obj.b;
        end

        function f = cost_batch(obj, w, indices)
            error('Not implemted yet.');        
        end

        function g = full_grad(obj, w)
            %g = obj.A' * (obj.A * w - obj.obj.b);
            g = obj.AtA * w - obj.Atb;
        end

        function g = grad(obj, w, indices)
            A_partial = obj.A(:,indices);
            g = A_partial' * (A_partial * w - obj.b);        
        end
        
        % for Subgradient descent
        function subg = full_subgrad(obj, w)
            subg = obj.lambda * sign(w);
        end
        
        % for Smoothing 
        function smooth_g = full_smooth_grad(obj, w)
            smooth_g = zeros(obj.n, 1);
            
            if strcmp(obj.smooth_type, 'type1') % f_mu(w) = w^2/2mu for abs(w)<= mu, abs(w)-mu/2 for abs(w)> mu.
                for i = 1 : obj.n
                    if abs(w(i)) <= obj.smooth_mu
                        smooth_g(i) = w(i) / obj.smooth_mu;
                    elseif w(i) > obj.smooth_mu
                        smooth_g(i) =1;
                    else
                        smooth_g(i) =-1;
                    end
                end
            elseif strcmp(obj.smooth_type, 'type2') % f_mu(w) = (w^2+mu^2)^(1/2)
                for i = 1 : obj.n 
                    smooth_g(i) = w(i)/sqrt(w(i)^2+obj.smooth_mu^2);
                end   
            elseif strcmp(obj.smooth_type, 'type3') % f_mu(w) = mu log((exp(-w/mu) + exp(w/mu))/2)
                for i = 1 : obj.n 
                    p = exp(-w(i)/obj.smooth_mu);                    
                    q = exp(w(i)/obj.smooth_mu);
                    smooth_g(i) = (q-p)/(p+q);
                end   
                %smooth_g = smooth_g;
            end
            
            smooth_g = obj.lambda * smooth_g;
        end        

        % for Frank-Wolfe, a.k.a. Condtional Gradient
        function [s, idx, sign_flag] = LMO(obj, grad) 
            [val, idx] = max( abs(grad) );
            obj.sign_flag = -1 * sign(grad(idx));
            obj.idx = idx;

            s = zeros(size(grad)); 
            s(idx) = obj.sign_flag * obj.lambda; 
            
            sign_flag = obj.sign_flag;
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
            As = sign_flag * obj.lambda * obj.A(:,idx); % = the i-th column of the dictionary matrix A
            As_minus_Ax = As - obj.A*w;
            step = max(0, min(1, As_minus_Ax' * (obj.b-obj.A*w) / (As_minus_Ax' * As_minus_Ax)));
        end          

        function w_opt = calc_solution(obj, method, options_in)

            if nargin < 1
                method = 'ista';
            end        

            options.max_epoch = options_in.max_epoch;
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
            elseif strcmp(method, 'ista')
                options.step_alg = 'fix';     
                options.step_init = 1/obj.L; 
                options.sub_mode  = 'FISTA';                
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

