classdef linear_regression
% This file defines l2-regularized linear regression problem class
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
%           f_i(w) = 1/2 * (w' * x_i - y_i)^2 + lambda/2 * w^2.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on March 24, 2018

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
        x_norm;
        x;         
    end
    
    methods
        function obj = linear_regression(x_train, y_train, x_test, y_test, varargin) 
            
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

            obj.name = 'linear_regression';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.hessain_w_independent = true;
            obj.x_norm = sum(x_train.^2,1);
            obj.x = x_train;    
        end

    
        function f = cost(obj, w)

            f = sum((w'*obj.x_train-obj.y_train).^2)/ (2 * obj.n_train) + obj.lambda/2*w'*w;

        end

        function g = grad(obj, w, indices)

            residual = w'*obj.x_train(:,indices)-obj.y_train(indices);
            g = obj.x_train(:,indices) * residual'/length(indices)+ obj.lambda*w;

        end

        function g = full_grad(obj, w)

            g = obj.grad(w, 1:obj.n_train);

        end

        function g = ind_grad(obj, w, indices)

            residual = w'*obj.x_train(:,indices)-obj.y_train(indices);
            g = obj.x_train(:,indices) * diag(residual) + obj.lambda* repmat(w, [1 length(indices)]);

        end

        function h = hess(obj, w, indices)
    %         % original code
    %         h = 0;
    %         len = length(indices);
    %         for ii=1:len
    %             idx = indices(ii);
    %             xx = x_train(:,indices(:,idx));
    %             h = h + xx * xx';
    %         end
    %         h = h/len + lambda * eye(d);

            h = 1/length(indices) * obj.x_train(:,indices) * (obj.x_train(:,indices)') + obj.lambda * eye(obj.d);
        end

        function h = full_hess(obj, w)

            h = hess(w, 1:obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            hv = 1/length(indices) * obj.x_train(:,indices) * ((obj.x_train(:,indices)'*v)) + obj.lambda*v;

        end

        function p = prediction(obj, w)
            p = w' * obj.x_test;        
        end

        function e = mse(obj, y_pred)

            e = sum((y_pred-obj.y_test).^2)/ (2 * obj.n_test);

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


        %% for Sub-sampled Newton
        function h = diag_based_hess(obj, w, indices, square_hess_diag)
            X = obj.x_train(:,indices)';
            h = X' * diag(square_hess_diag) * X /length(indices) + obj.lambda * eye(obj.d);
        end  

        function square_hess_diag = calc_square_hess_diag(obj, w, indices)
            len = nnz(indices);
            square_hess_diag = ones(len,1);
        end  
    end

end

