classdef binary_linear_classification
% This file defines binary linear classification problem (with least squares) class
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn, y_i = {0,1}.
%       lambda      l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda/2 * w^2.
%           f_i(w) = (y_i - phi(<w, x_i>) )^2, 
%           where phi(z) = 1 / (1+exp(-z)).
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 19, 2018


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
        function obj = binary_linear_classification(x_train, y_train, x_test, y_test, varargin)    
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            

            if nargin < 5
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end

            obj.d = size(obj.x_train, 1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);      
            obj.name = 'binary_linear_classification';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.lambda = obj.lambda;
            obj.classes = 2;  
            obj.hessain_w_independent = false;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end

        function f = cost(obj, w)
    %         The above is replaced with below by HK on 2017/12/5
            %sigmod_result = sigmoid(obj.y_train.*(w'*obj.x_train));
            %sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            %f = -sum(log(sigmod_result),2)/obj.n_train + obj.lambda * (w'*w) / 2;
            
            f = mean((obj.y_train' - sigmoid(obj.x_train' * w)).^2) + 0.5 * obj.lambda * norm(w)^2;        
        end
        
        function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2) + obj.lambda * (w'*w) / 2;

        end

        function g = grad(obj, w, indices)
            
            n = length(indices);
            X_sub = obj.x_train(:,indices);
            y_sub = obj.y_train(indices);

            a = sigmoid(X_sub' * w);
            g = 2 * X_sub *((a - y_sub').*a.*(1-a)) + obj.lambda * w;
            g = g / n;
        end

        function g = full_grad(obj, w)

            a = sigmoid(obj.x_train' * w);
            g = 2 * obj.x_train * ((a - obj.y_train').*a.*(1-a)) + obj.lambda * w;
            g = g / obj.n_train;
        end

        function g = ind_grad(obj, w, indices)

            g = -ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * diag(ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))+ obj.lambda* repmat(w, [1 length(indices)]);

        end

%         function h = hess(obj, w, subsamp_size)
%             
%             XY = datasample([obj.x_train', obj.y_train'], subsamp_size, 1, 'Replace', false);
%             X_sub = XY(:,1:end-1);
%             y_sub = XY(:,end);
% 
%             n = length(y_sub);
%             a = sigmoid(X_sub * w);
%             c = 2/n * ((a - y_sub).*a.*(1-a).*(1 - 2*a) + a.^2 .*(1 - a).^2);  
%             
%             h = X_sub'*bsxfun(@times,c,X_sub)+ obj.lambda * eye(obj.d);
%             h = (h + h')/2;            
% 
%         end
        
        function h = hess(obj, w, indices)
            
            n = length(indices);
            
            X_sub = obj.x_train(:,indices)';
            y_sub = obj.y_train(indices)';
            
            a = sigmoid(X_sub * w);
            c = 2/n * ((a - y_sub).*a.*(1-a).*(1 - 2*a) + a.^2 .*(1 - a).^2);  
            
            h = X_sub'*bsxfun(@times,c, X_sub)+ obj.lambda * eye(obj.d);
            h = (h + h')/2;            

        end        

        function h = full_hess(obj, w)

            h = hess(obj, w, obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            n = length(indices);
            
            X_sub = obj.x_train(:,indices)';
            y_sub = obj.y_train(indices)';
            
            a = sigmoid(X_sub * w);
            c = 2/n * ((a - y_sub).*a.*(1-a).*(1 - 2*a) + a.^2 .*(1 - a).^2);  
            
            h = X_sub'*bsxfun(@times,c, X_sub)+ obj.lambda * eye(obj.d);
            h = (h + h')/2;   
            
            hv = h * v;

        end

        function p = prediction(obj, w, mode)
            if strcmp(mode, 'train')
                p = sigmoid(obj.x_train*(w)) > 0.5;
            else
                p = sigmoid(obj.x_test*(w)) > 0.5;
            end
        end
        
        function p_all = prediction_all(obj, w_array, mode)
            len = size(w_array, 2);
            if strcmp(mode, 'train')            
                p_all = zeros(obj.n_train, len);
            else
                p_all = zeros(obj.n_test, len);
            end
            
            for i = 1 : len
                if strcmp(mode, 'train')
                    p_all(:, i) = sigmoid(obj.x_train'*(w_array(:,i))) > 0.5;
                else
                    p_all(:, i) = sigmoid(obj.x_test'*(w_array(:,i))) > 0.5;
                end
            end
        end        

        function a = accuracy(obj, y_pred, mode)
            if strcmp(mode, 'train')
                a = bsxfun(@(x,y)x==y, y_pred, obj.y_train);
            else
                a = bsxfun(@(x,y)x==y, y_pred, obj.y_test);               
            end
        end
        
        function a_all = accuracy_all(obj, y_pred_array, mode)
            len = size(y_pred_array, 2);
            a_all = zeros(len, 1);
            
            for i = 1 : len 
                if strcmp(mode, 'train')
                    corrects = bsxfun(@(x,y)x==y, y_pred_array(:, i), obj.y_train');
                else
                    corrects = bsxfun(@(x,y)x==y, y_pred_array(:, i), obj.y_test');
                end
                a_all(i) = mean(corrects);
            end
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


        %% for NIM
        function [labels, samples] = get_partial_samples(obj, indices)
            samples = obj.x_train(:,indices);
            labels  = obj.y_train(indices);
        end

        function [s] = phi_prime(obj, w, indices)
            e = exp(-1.0 * obj.y_train(indices)' .* (obj.x_train(:,indices)'*w));
            s = e ./ (1.0+e);        
        end

        function [ss] = phi_double_prime(obj, w, indices)
            e = exp(-1.0 * obj.y_train(indices)' .* (obj.x_train(:,indices)'*w));
            s = e ./ (1.0+e); 
            ss = s .* (1.0 - s);
        end


        %% for Sub-sampled Newton
        function h = diag_based_hess(obj, w, indices, square_hess_diag)
            X = obj.x_train(:,indices)';
            h = X' * diag(square_hess_diag) * X / length(indices) + obj.lambda * eye(obj.d);
        end  

        function square_hess_diag = calc_square_hess_diag(obj, w, indices)
            %hess_diag = 1./(1+exp(Y.*(X*w)))./(1+exp(-Y.*(X*w)));
            Xw = obj.x_train(:,indices)'*w;
            y = obj.y_train(indices)';
            yXw = y .* Xw;
            square_hess_diag = 1./(1+exp(yXw))./(1+exp(-yXw));
        end    

    end
end

