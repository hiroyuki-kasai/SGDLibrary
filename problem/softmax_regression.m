function Problem = softmax_regression(x_train, y_train, x_test, y_test, lambda, num_class)
% This file defines softmax regression (multinomial logistic regression) problem.
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
%           P(y_i=l|x_i,w) = w' * x_i - log (sum_{l=1}^n_classes exp(w' * x_i)).
%
% "w" is the 'vectorized' model parameter of size d x num_class matrix.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on Oct. 25, 2016


    d = size(x_train, 1);
    n_classes = num_class;
    n_train = length(y_train);    
    n_test = length(y_test);
    
    Problem.name = @() 'softmax_regression';
    Problem.dim = @() d*n_classes;
    Problem.samples = @() n_train;
    Problem.classes = @() n_classes;   
    Problem.hessain_w_independent = @() false;

    Problem.cost = @cost;
    function f = cost(w)
        
        w_mat = reshape(w, [d n_classes]);
        
        % calculate log P(y_train=l|X,W) = W'X - log (sum_{l=1}^n_classes exp(W'X))
        log_p = w_mat'*x_train - ones(n_classes, 1) * logsumexp(w_mat'*x_train); 

        % calculate 1/N sum_n sum_l I(y_train=l) log P(y_train=l|X,W)
        y_train = logical(y_train);        
        logprob = sum(log_p(y_train))/n_train; 
        
        % calculate cost
        f = -logprob + lambda * (w'*w) / 2;
        
    end

    Problem.grad = @grad;
    function g = grad(w, indices)
        
        w_mat = reshape(w, [d n_classes]);

        % calculate log P(y_train=l|X,W) = W'X - log (sum_{l=1}^n_classes exp(W'X))        
        log_p = w_mat'*x_train(:,indices) - ones(n_classes, 1) * logsumexp(w_mat'*x_train(:,indices)); 
        % calculate  1{y_train=l} - P(y_train=l|X,W) because exp(log(P))=P.
        p = y_train(:,indices) - exp(log_p);
        % calculate x_train p'
        g = x_train(:,indices) * p';
        g = g(:) / length(indices);
        g = -g + lambda * w;
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        
        g = grad(w, 1:n_train);
        
    end

    Problem.hess = @hess; % To Do
    function h = hess(w, indices)
        
        warning('not supported');
        
    end

    Problem.hess_vec = @hess_vec; % To Do
    function hv = hess_vec(w, v, indices)
        
        warning('not supported');
        
    end

    Problem.prediction = @prediction;
    function max_class = prediction(w)
        
        w_mat = reshape(w, [d n_classes]);        
        p = w_mat' * x_test;
        [~, max_class] = max(p, [], 1);
        
    end

    Problem.accuracy = @accuracy;
    function a = accuracy(class_pred)
        
        [~, class_test] = max(y_test, [], 1);
        a = sum(class_pred == class_test) / n_test; 
        
    end

    Problem.calc_solution = @calc_solution;
    function w_star = calc_solution(problem, maxiter)
        
        options.max_iter = maxiter;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;        
        options.tol_gnorm = 1.0e-16;    
        options.step_alg = 'backtracking';        
        [w_star,~] = gd(problem, options);
        
    end

end

