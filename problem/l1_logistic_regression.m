classdef l1_logistic_regression
% This file defines logistic regression (binary classifier) problem class with l1-norm
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       varargin    options.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda || w ||_1.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 18, 2017
% Modified by H.Kasai on Mar. 25, 2017
%

    properties
        name;    
        dim;
        samples;
        lambda;
        classes;  
        hessain_w_independent;
        d;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test;
        x_norm;
        x; 
        prox_flag;        
    end
    
    methods
        function obj = l1_logistic_regression(x_train, y_train, x_test, y_test, varargin)
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            

            if nargin < 5
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end
            
            if obj.lambda > 0
                obj.prox_flag = true;
            else
                obj.prox_flag = false;
            end

            obj.d = size(obj.x_train, 1);
            obj.n_train = length(obj.y_train);
            obj.n_test = length(obj.y_test);      

            obj.name = 'l1 logistic_regression';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;  
            obj.hessain_w_independent = false;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end


        function v = prox(obj, w, t) % l1 soft threshholding
            v = soft_thresh(w, t * obj.lambda);
        end     

        function f = cost(obj, w)

            f = -sum(log(sigmoid(obj.y_train.*(w'*obj.x_train))),2)/obj.n_train + obj.lambda * norm(w,1);

        end

        function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2) + obj.lambda * norm(w,1);

        end

        % calculate l1 norm
        function r = reg(obj, w)
            r = norm(w,1);
        end

        function g = grad(obj, w, indices)

            e = exp(-1*obj.y_train(indices)'.*(obj.x_train(:,indices)'*w));
            s = e./(1+e);
            g = -(1/length(indices))*((s.*obj.y_train(indices)')'*obj.x_train(:,indices)')';
            g = full(g);

        end

        function g = full_grad(obj, w)

            g = obj.grad(w, 1:obj.n_train);

        end

        function h = hess(obj, w, indices)

            %org code
            %temp = exp(-1*(y_train(indices)').*(x_train(:,indices)'*w));
            %b = temp ./ (1+temp);
            %h = 1/length(indices)*x_train(:,indices)*(diag(b-b.^2)*(x_train(:,indices)')); 

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            h = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * obj.x_train(:,indices)';
        end

        function h = full_hess(obj, w)

            h = obj.hess(w, 1:obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            hv = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * (obj.x_train(:,indices)' * v);

        end

        function p = prediction(obj, w)

            p = sigmoid(w' * obj.x_test);

            class1_idx = p>0.5;
            class2_idx = p<=0.5;         
            p(class1_idx) = 1;
            p(class2_idx) = -1;         

        end

        function a = accuracy(obj, y_pred)

            a = sum(y_pred == obj.y_test) / obj.n_test; 

        end

        function w_opt = calc_solution(obj, options_in, method)

            if nargin < 3
                method = 'ag';
            end        

            options.max_iter = options_in.max_iter;
            options.w_init = options_in.w_init;
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
            elseif strcmp(method, 'ag')
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_opt,~] = ag(obj, options);            
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end
        end
    end
end

