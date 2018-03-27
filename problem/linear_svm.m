classdef linear_svm
% This file defines l2-regularized SVM problem
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda      l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = 1/2 * (max(0.0, 1 - y_i .* (w'*x_i) )^2 + lambda/2 * w^2.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on Mar. 23, 2018

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
    end  
    
    methods
        function obj = linear_svm(x_train, y_train, x_test, y_test, varargin)
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;    
            
            if nargin < 5
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end            
            
            obj.d = size(x_train, 1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);  

            obj.name = 'linear_svm';
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;   
            obj.hessain_w_independent = false; 
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;             
        end
  

        function f = cost(obj, w)    

            f_sum = 0.5 * sum(max(0.0, 1 - obj.y_train' .*(w'*obj.x_train)').^2);
            f = f_sum/obj.n_train + obj.lambda/2 * w(:)'*w(:);

        end

        function g = grad(obj, w, indices)

            alpha = w' * obj.x_train(:,indices);
            flag = obj.y_train(indices) .* alpha;
            flag(flag<1.0) = 1;
            flag(flag>1.0) = 0;

            coeff = flag' .* (1 - obj.y_train(indices)' .* alpha') .* obj.y_train(indices)';
            coeff = coeff';
            coeff = repmat(coeff,[obj.d 1]);
            g = obj.lambda * w - sum(coeff .* obj.x_train(:,indices),2)/length(indices);  

        end

        function g = full_grad(obj, w)

            g = obj.grad(w, 1:obj.n_train);

        end

        function h = hess(obj, w, indices)

            alpha = w' * obj.x_train(:,indices);
            flag = obj.y_train(indices) .* alpha;
            flag_indices = flag<1.0;

            x_part_new = obj.x_train(:,indices);
            x_part_new = x_part_new(:,flag_indices);

            h = obj.lambda * eye(obj.d) + x_part_new * x_part_new'/length(indices); 
        end

        function hv = hess_vec(obj, w, v, indices)

            alpha = w' * obj.x_train(:,indices);
            flag = obj.y_train(indices) .* alpha;
            flag_indices = flag<1.0;

            x_part_new = obj.x_train(:,indices);
            x_part_new = x_part_new(:,flag_indices);

            hv = obj.lambda * v + (x_part_new * (x_part_new'*v) )/length(indices); 
        end

        function p = prediction(obj, w)

            p = w' * obj.x_test;

            class1_idx = p>0;
            class2_idx = p<=0;         
            p(class1_idx) = 1;
            p(class2_idx) = -1;        

        end

        function a = accuracy(obj, y_pred)

            a = sum(y_pred == obj.y_test) / obj.n_test; 

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

