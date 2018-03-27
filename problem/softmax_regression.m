classdef softmax_regression
% This file defines softmax regression (multinomial logistic regression) problem class.
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda      l2-regularized parameter. 
%       num_class   number of classes. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = - sum_l I(y_i=l) log P(y_i=l|x_i,w) + lambda/2 * w^2,
%           where 
%           P(y_i=l|x_i,w) = w' * x_i - log (sum_{l=1}^num_class exp(w' * x_i)).
%
% "w" is the 'vectorized' model parameter of size d x num_class matrix.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on Oct. 25, 2016


    properties
        name;    
        dim;
        samples;
        lambda;
        hessain_w_independent;
        d;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test; 
        num_class;
        x_norm;
        x;
        classes;
    end
    
    methods    
    
        function obj = softmax_regression(x_train, y_train, x_test, y_test, num_class, varargin)    
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test; 
            obj.num_class = num_class;
            
            if nargin < 6
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end 
            
            obj.d = size(x_train, 1);
            obj.n_train = length(obj.y_train);    
            obj.n_test = length(obj.y_test);            

            obj.name = 'softmax_regression';
            obj.dim = obj.d * obj.num_class;
            obj.samples = obj.n_train;
            obj.hessain_w_independent = false;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train; 
            obj.classes = obj.num_class;
        end

        function f = cost(obj, w)

            w_mat = reshape(w, [obj.d obj.num_class]);

            % calculate log P(y_train=l|X,W) = W'X - log (sum_{l=1}^num_class exp(W'X))
            log_p = w_mat'*obj.x_train - ones(obj.num_class, 1) * logsumexp(w_mat'*obj.x_train); 

            % calculate 1/N sum_n sum_l I(y_train=l) log P(y_train=l|X,W)
            y_train_new = logical(obj.y_train);        
            logprob = sum(log_p(y_train_new))/obj.n_train; 

            % calculate cost
            f = -logprob + obj.lambda * (w'*w) / 2;

        end

        function g = grad(obj, w, indices)

            w_mat = reshape(w, [obj.d obj.num_class]);

            % calculate log P(y_train=l|X,W) = W'X - log (sum_{l=1}^num_class exp(W'X))        
            log_p = w_mat'*obj.x_train(:,indices) - ones(obj.num_class, 1) * logsumexp(w_mat'*obj.x_train(:,indices)); 
            % calculate  1{y_train=l} - P(y_train=l|X,W) because exp(log(P))=P.
            p = obj.y_train(:,indices) - exp(log_p);
            % calculate x_train p'
            g = obj.x_train(:,indices) * p';
            g = g(:) / length(indices);
            g = -g + obj.lambda * w;

        end

        function g = full_grad(obj, w)

            g = obj.grad(w, 1:obj.n_train);

        end

        function h = hess(obj, w, indices) % To Do

            warning('not supported');

        end

        function hv = hess_vec(obj, w, v, indices) % To Do

            warning('not supported');

        end

        function max_class = prediction(obj, w)

            w_mat = reshape(w, [obj.d obj.num_class]);        
            p = w_mat' * obj.x_test;
            [~, max_class] = max(p, [], 1);

        end

        function a = accuracy(obj, class_pred)

            [~, class_test] = max(obj.y_test, [], 1);
            a = sum(class_pred == class_test) / obj.n_test; 

        end

        function w_opt = calc_solution(obj, maxiter, method)
            
            if nargin < 3
                method = 'lbfgs';
            end                   

            options.max_iter = maxiter;
            options.verbose = true;
            options.tol_optgap = 1.0e-24;        
            options.tol_gnorm = 1.0e-16;   
            
            options.step_alg = 'backtracking';        
            if strcmp(method, 'sd')
                [w_opt,~] = sd(obj, options);
            elseif strcmp(method, 'cg')
                [w_opt,~] = ncg(obj, options);
            elseif strcmp(method, 'newton')
                options.sub_mode = 'INEXACT';    
                options.step_alg = 'non-backtracking'; 
                [w_opt,~] = newton(obj, options);
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end

        end
    end

end

