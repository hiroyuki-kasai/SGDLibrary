function Problem = linear_svm(x_train, y_train, x_test, y_test, lambda)
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
% Modified by H.Kasai on Oct. 26, 2016


    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);  
    
    Problem.name = @() 'linear_svm';
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.classes = @() 2;   
    Problem.hessain_w_independent = @() false;

    Problem.cost = @cost;
    function f = cost(w)
        
        f_sum = 0.5 * sum(max(0.0, 1 - y_train' .*(w'*x_train)').^2);
        f = f_sum/n_train + lambda/2 * w(:)'*w(:);
        
    end

    Problem.grad = @grad;
    function g = grad(w, indices)

        alpha = w' * x_train(:,indices);
        flag = y_train(indices) .* alpha;
        flag(flag<1.0) = 1;
        flag(flag>1.0) = 0;

        coeff = flag' .* (1 - y_train(indices)' .* alpha') .* y_train(indices)';
        coeff = coeff';
        coeff = repmat(coeff,[d 1]);
        g = lambda * w - sum(coeff .* x_train(:,indices),2)/length(indices);  

    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)

        g = grad(w, 1:n_train);

    end

    Problem.hess = @hess;
    function h = hess(w, indices)
        
        alpha = w' * x_train(:,indices);
        flag = y_train(indices) .* alpha;
        flag_indices = flag<1.0;
        
        x_part_new = x_train(:,indices);
        x_part_new = x_part_new(:,flag_indices);
        
        h = lambda * eye(d) + x_part_new * x_part_new'/length(indices); 
    end

    Problem.hess_vec = @hess_vec;
    function hv = hess_vec(w, v, indices)
        
        alpha = w' * x_train(:,indices);
        flag = y_train(indices) .* alpha;
        flag_indices = flag<1.0;
        
        x_part_new = x_train(:,indices);
        x_part_new = x_part_new(:,flag_indices);
        
        hv = lambda * v + (x_part_new * (x_part_new'*v) )/length(indices); 
    end

    Problem.prediction = @prediction;
    function p = prediction(w)
        
        p = w' * x_test;
        
        class1_idx = p>0;
        class2_idx = p<=0;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;        
        
    end

    Problem.accuracy = @accuracy;
    function a = accuracy(y_pred)
        
        a = sum(y_pred == y_test) / n_test; 
        
    end

    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, maxiter)
        
        options.max_iter = maxiter;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;        
        options.tol_gnorm = 1.0e-16;
        options.step_alg = 'backtracking';        
        [w_opt,~] = gd(problem, options);
        
    end

end

