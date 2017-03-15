function Problem = linear_regression(x_train, y_train, x_test, y_test, lambda)
% This file defines l2-regularized linear regression problem
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
% Modified by H.Kasai on Oct. 25, 2016


    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);      
   
    Problem.name = @() 'linear_regression';    
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.hessain_w_independent = @() true;

    Problem.cost = @cost;
    function f = cost(w)

        f = sum((w'*x_train-y_train).^2)/ (2 * n_train) + lambda/2*w'*w;
        
    end

    Problem.grad = @grad;
    function g = grad(w, indices)

        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * residual'/length(indices)+ lambda*w;
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)

        g = grad(w, 1:n_train);
        
    end

    Problem.ind_grad = @ind_grad;
    function g = ind_grad(w, indices)
 
        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * diag(residual) + lambda* repmat(w, [1 length(indices)]);
         
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
        
        h = 0.5/length(indices) * x_train(:,indices) * (x_train(:,indices)') + lambda * eye(d);
        
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        
        h = hess(w, 1:n_train);
        
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        
        hv = 0.5/length(indices) * x_train(:,indices) * ((x_train(:,indices)'*v)) + lambda*v;
        
    end

    Problem.prediction = @prediction;
    function p = prediction(w)
        p = w' * x_test;        
    end

    Problem.mse = @mse;
    function e = mse(y_pred)
        
        e = sum((y_pred-y_test).^2)/ (2 * n_test);
        
    end

    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, maxiter)
        
        options.max_iter = maxiter;
        options.verbose = true;
        options.tol_optgap = 1.0e-16;        
        options.tol_gnorm = 1.0e-16;  
        options.step_alg = 'backtracking';
        [w_opt,~] = gd(problem, options);
        %[w_opt,~] = ncg(problem, options);
        
    end

end

