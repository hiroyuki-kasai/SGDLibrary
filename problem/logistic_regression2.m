classdef logistic_regression2
% This file defines logistic regression (binary classifier) problem class (version 2)
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

%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 03, 2018


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
        function obj = logistic_regression2(x_train, y_train, x_test, y_test, varargin)    
            
            obj.x_train = x_train';     % transpose for efficiency
            obj.y_train = y_train;
            obj.x_test = x_test';       % transpose for efficiency
            obj.y_test = y_test;            

            if nargin < 5
                obj.lambda = 0.1;
            else
                obj.lambda = varargin{1};
            end

            obj.d = size(obj.x_train, 2);
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
            z = obj.x_train * w;
            f = - ((log_phi(z)'*obj.y_train') + (ones(obj.n_train,1)-obj.y_train')'*(one_minus_log_phi(z))) / obj.n_train;  
            f = f + 0.5 * obj.lambda * (norm(w).^2);
        end
        
        function f = cost_batch(obj, w, indices)

            % not implemented

        end

        function g = grad(obj, w, indices)

            X = obj.x_train(indices,:);
            z = X * w; 
            h = phi(z);
            g = X'*(h-obj.y_train(indices)')/length(indices);
            g = g + obj.lambda *w;
            
        end

        function g = full_grad(obj, w)

            g = grad(obj, w, 1:obj.n_train);
        end

        function g = ind_grad(obj, w, indices)

            % not implemented

        end

        function h = hess(obj, w, indices)

            n_size = length(indices);
            X = obj.x_train(indices,:);
            z = X * w;
            q = phi(z);
            h = q .* (ones(n_size,1) - phi(z));
            %H = (X'*(h .* X)) / n_size; % for only new MATLAB
            H = (X'*(repmat(h,1,obj.d) .* X)) / n_size; % can be calculated in old MATLAB

            H = H + obj.lambda * eye(obj.d, obj.d);            

            h = H;
        end

        function h = full_hess(obj, w)

            h = hess(obj, w, 1:obj.n_train);

        end

        function hv = hess_vec(obj, w, v, indices)

            n_size = length(indices);
            X = obj.x_train(indices,:);         
            z = X * w;
            z = phi(-z);
            d_binary = z.*(ones(n_size,1) - z);
            wa = d_binary .* (X * v);
            Hv = X'* wa / n_size;
            hv = Hv + obj.lambda * v;

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
            %samples = obj.x_train(:,indices);
            %labels  = obj.y_train(indices);
        end

        function [s] = phi_prime(obj, w, indices)
            %e = exp(-1.0 * obj.y_train(indices)' .* (obj.x_train(:,indices)'*w));
            %s = e ./ (1.0+e);        
        end

        function [ss] = phi_double_prime(obj, w, indices)
            %e = exp(-1.0 * obj.y_train(indices)' .* (obj.x_train(:,indices)'*w));
            %s = e ./ (1.0+e); 
            %ss = s .* (1.0 - s);
        end


        %% for Sub-sampled Newton
        function h = diag_based_hess(obj, w, indices, square_hess_diag)
            
            %X = obj.x_train(:,indices)';
            %h = X' * diag(square_hess_diag) * X / length(indices) + obj.lambda * eye(obj.d);
            
        end  

        function square_hess_diag = calc_square_hess_diag(obj, w, indices)

            %Xw = obj.x_train(:,indices)'*w;
            %y = obj.y_train(indices)';
            %yXw = y .* Xw;
            %square_hess_diag = 1./(1+exp(yXw))./(1+exp(-yXw));
        end    

    end
end


function out = phi(t) % Author: Fabian Pedregosa
    % logistic function returns 1 / (1 + exp(-t))
    idx = t>0;
    [t_size, ~] = size(t);
    one = ones(t_size,1);
    out = zeros(t_size,1);
    out(idx) = 1.0 ./ (1 + exp(-t(idx)));
    exp_t = zeros(t_size,1);
    exp_t(~idx) = exp(t(~idx));
    out(~idx) = exp_t(~idx) ./ (one(~idx) + exp_t(~idx));
end

function out = log_phi(t)
    % log(Sigmoid): log(1 / (1 + exp(-t)))
    idx = t>0;
    [t_size, ~] = size(t);
    out = zeros(t_size,1);
    out(idx) = -log(1 + exp(-t(idx)));
    out(~idx) = t(~idx) - log(1 + exp(t(~idx)));
end


function out = one_minus_log_phi(t)
    % log(1-Sigmoid): log(1-1 / (1 + exp(-t)))
    idx = t>0;
    [t_size, ~] = size(t);
    out = zeros(t_size,1);
    out(idx) = -t(idx) -log(1 + exp(-t(idx)));
    out(~idx) = -log(1 + exp(t(~idx)));
end

