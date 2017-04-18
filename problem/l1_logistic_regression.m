function [Problem] = l1_logistic_regression(x_train, y_train, x_test, y_test, varargin)
% This file defines logistic regression (binary classifier) problem with l1-norm.
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
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda || w ||_1.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 18, 2017
%

    if nargin < 5
        lambda = 0.1;
    else
        lambda = varargin{1};
    end

    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);      
    
    Problem.name = @() 'l1 logistic_regression';    
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.lambda = @() lambda;
    Problem.classes = @() 2;  
    Problem.hessain_w_independent = @() false;
    Problem.x_norm = @() sum(x_train.^2,1);
    Problem.x = @() x_train;
    
    
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda);
    end     

    Problem.cost = @cost;
    function f = cost(w)

        f = -sum(log(sigmoid(y_train.*(w'*x_train))),2)/n_train + lambda * norm(w,1);
        
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        
        f = -sum(log(sigmoid(y_train(indices).*(w'*x_train(:,indices))))/n_train,2) + lambda * norm(w,1);
        
    end

    % calculate l1 norm
    Problem.reg = @reg;
    function r = reg(w)
        r = norm(w,1);
    end

    Problem.grad = @grad;
    function g = grad(w, indices)

        e = exp(-1*y_train(indices)'.*(x_train(:,indices)'*w));
        s = e./(1+e);
        g = -(1/length(indices))*((s.*y_train(indices)')'*x_train(:,indices)')';
        g = full(g);
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)

        g = grad(w, 1:n_train);
    
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
        
        %org code
        %temp = exp(-1*(y_train(indices)').*(x_train(:,indices)'*w));
        %b = temp ./ (1+temp);
        %h = 1/length(indices)*x_train(:,indices)*(diag(b-b.^2)*(x_train(:,indices)')); 
        
        sigm_val = sigmoid(y_train(indices).*(w'*x_train(:,indices)));
        c = sigm_val .* (ones(1,length(indices))-sigm_val); 
        h = 1/length(indices)* x_train(:,indices) * diag(y_train(indices).^2 .* c) * x_train(:,indices)';
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        
        h = hess(w, 1:n_train);
        
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        
        sigm_val = sigmoid(y_train(indices).*(w'*x_train(:,indices)));
        c = sigm_val .* (ones(1,length(indices))-sigm_val); 
        hv = 1/length(indices)* x_train(:,indices) * diag(y_train(indices).^2 .* c) * (x_train(:,indices)' * v);
        
    end

    %
    Problem.prediction = @prediction;
    function p = prediction(w)
        
        p = sigmoid(w' * x_test);
        
        class1_idx = p>0.5;
        class2_idx = p<=0.5;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;         
        
    end

    Problem.accuracy = @accuracy;
    function a = accuracy(y_pred)
        
        a = sum(y_pred == y_test) / n_test; 
        
    end

    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, maxiter, method)
        
        if nargin < 3
            method = 'lbfgs';
        end        
        
        options.max_iter = maxiter;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;
        options.tol_gnorm = 1.0e-16;
        options.step_alg = 'backtracking';
        
        if strcmp(method, 'sg')
            [w_opt,~] = gd(problem, options);
        elseif strcmp(method, 'cg')
            [w_opt,~] = ncg(problem, options);
        elseif strcmp(method, 'newton')
            options.sub_mode = 'INEXACT';    
            options.step_alg = 'non-backtracking'; 
            [w_opt,~] = newton(problem, options);
        else 
            options.step_alg = 'backtracking';  
            options.mem_size = 5;
            [w_opt,~] = lbfgs(problem, options);              
        end
    end



end

