classdef logistic_regression
% This file defines logistic regression (binary classifier) problem class
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
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda/2 * w^2.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on March 23, 2018


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
        function obj = logistic_regression(x_train, y_train, x_test, y_test, varargin)    
            
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
            obj.name = 'logistic_regression';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.lambda = obj.lambda;
            obj.classes = 2;  
            obj.hessain_w_independent = false;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end

        function f = cost(obj, w)
            %f_old = sum(log(1+exp(-y_train.*(w'*x_train)))/n_train,2)+ lambda * (w'*w) / 2; %is replaced below 
            % becasuse log(sigmoid(a)) = log(1/(1+exp(-a))) = log1 - log(1+exp(-a)) = -log(1+exp(-a)).
            %f = -sum(log(sigmoid(y_train.*(w'*x_train)))/n_train,2) + lambda * (w'*w) / 2;

    %         Commented out below due to avoid '-Inf' values of g by HK on 2017/12/5
    %         f = -sum(log(sigmoid(y_train.*(w'*x_train))),2)/n_train + lambda * (w'*w) / 2;

    %         The above is replaced with below by HK on 2017/12/5
            sigmod_result = sigmoid(obj.y_train.*(w'*obj.x_train));
            sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            f = -sum(log(sigmod_result),2)/obj.n_train + obj.lambda * (w'*w) / 2;
        end
        
        function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2) + obj.lambda * (w'*w) / 2;

        end

        function g = grad(obj, w, indices)

            % org code
            %g = -sum(((ones(obj.d,1) * (sigmoid(-obj.y_train(indices).*(w'*obj.x_train(:,indices))) .* obj.y_train(indices))) .* obj.x_train(:,indices))/length(indices),2) + obj.lambda * w;
            %
            % (log(1+exp(-y_train.*(w'*x_train)))' = -y_train.*x_train * (exp(-y_train.*(w'*x_train))/(1+exp(-y_train.*(w'*x_train))))
            %   = -y_train.*x_train * (1 - sigmoid(y_train.*(w'*x_train)))
            % or
            % (log(sigmoid(y_train.*(w'*x_train)))' = y_train.*x_train * 1/sigmoid(y_train.*(w'*x_train)) * (sigmoid(y_train.*(w'*x_train)))'
            %   = y_train.*x_train * 1/sigmoid(y_train.*(w'*x_train)) * (- sigmoid(y_train.*(w'*x_train)) * (1 - sigmoid(y_train.*(w'*x_train))))
            %   = -y_train.*x_train * (1 - sigmoid(y_train.*(w'*x_train)))

    %         Replace the commented-out lines with below (although it is a bit slower) due to avoid 'NAN' values of g by HK on 2017/12/5
            g = -sum(ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * (ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))',2)/length(indices)+ obj.lambda * w;

    %         Commented out below due to avoid 'NAN' values of g by HK on 2017/12/5
%             e = exp(-1*obj.y_train(indices)'.*(obj.x_train(:,indices)'*w));
%             s = e./(1+e);
%             g = -(1/length(indices))*((s.*obj.y_train(indices)')'*obj.x_train(:,indices)')';
%             g = full(g) + obj.lambda * w;

        end

        function g = full_grad(obj, w)

            %g = -sum(ones(d,1) * y_train.*x_train * (ones(1,n_train)-sigmoid(y_train.*(w'*x_train)))',2)/n_train+ lambda * w;
            g = grad(obj, w, 1:obj.n_train);
        end

        function g = ind_grad(obj, w, indices)

            g = -ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * diag(ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))+ obj.lambda* repmat(w, [1 length(indices)]);

        end

        function h = hess(obj, w, indices)

            %org code
            %temp = exp(-1*(y_train(indices)').*(x_train(:,indices)'*w));
            %b = temp ./ (1+temp);
            %h = 1/length(indices)*x_train(:,indices)*(diag(b-b.^2)*(x_train(:,indices)'))+lambda*eye(d); 

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            h = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * obj.x_train(:,indices)'+obj.lambda*eye(obj.d);

        end

        function h = full_hess(obj, w)

            h = hess(obj, w, 1:obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            hv = 1/length(indices)* obj.x_train(:,indices) * diag(obj.y_train(indices).^2 .* c) * (obj.x_train(:,indices)' * v) +obj.lambda*v;

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

