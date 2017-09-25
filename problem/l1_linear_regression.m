function Problem = l1_linear_regression(x_train, y_train, x_test, y_test, lambda)
% This file defines l1-regularized linear regression problem
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
%           f_i(w) = 1/2 * (w' * x_i - y_i)^2 + || w ||_1.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Sep. 25, 2017


    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);      
   
    Problem.name = @() 'linear_regression';    
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.lambda = @() lambda;    
    Problem.hessain_w_independent = @() true;
    Problem.x_norm = @() sum(x_train.^2,1);
    Problem.x = @() x_train;    

    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
        v = soft_thresh(w, t * lambda);
    end       
    
    Problem.cost = @cost;
    function f = cost(w)

        f = sum((w'*x_train-y_train).^2)/ (2 * n_train) + lambda * norm(w,1);
        
    end

    Problem.cost_batch = @cost_batch;
    function f = cost_batch(w, indices)
        
        f = sum((w'*x_train-y_train).^2)/ (2 * n_train) + lambda * norm(w,1);
        
    end

    % calculate l1 norm
    Problem.reg = @reg;
    function r = reg(w)
        r = norm(w,1);
    end

    Problem.grad = @grad;
    function g = grad(w, indices)

        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * residual'/length(indices);
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)

        g = grad(w, 1:n_train);
        
    end

    Problem.ind_grad = @ind_grad;
    function g = ind_grad(w, indices)
 
        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * diag(residual);
         
    end

    Problem.hess = @hess; 
    function h = hess(w, indices)
%         % original code
%         h = 0;
%         len = length(indices);
%         for ii=1:len
%             idx = indices(ii);
%             xx = x_train(:,indices(:,idx));
%             h = h + xx * xx';
%         end
%         h = h/len + lambda * eye(d);
        
        h = 1/length(indices) * x_train(:,indices) * (x_train(:,indices)');
    end

    Problem.full_hess = @full_hess; 
    function h = full_hess(w)
        
        h = hess(w, 1:n_train);
        
    end

    Problem.hess_vec = @hess_vec; 
    function hv = hess_vec(w, v, indices)
        
        hv = 1/length(indices) * x_train(:,indices) * ((x_train(:,indices)'*v));
        
    end

    Problem.prediction = @prediction;
    function p = prediction(w)
        p = w' * x_test;        
    end

    Problem.mse = @mse;
    function e = mse(y_pred)
        
        e = sum((y_pred-y_test).^2)/ (2 * n_test);
        
    end


    %% for Sub-sampled Newton
    Problem.diag_based_hess = @diag_based_hess;
    function h = diag_based_hess(w, indices, square_hess_diag)
        X = x_train(:,indices)';
        h = X' * diag(square_hess_diag) * X /length(indices);
    end  

    Problem.calc_square_hess_diag = @calc_square_hess_diag;
    function square_hess_diag = calc_square_hess_diag(w, indices)
        len = nnz(indices);
        square_hess_diag = ones(len,1);
    end 



    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, method, options_in)
        
        if nargin < 2
            method = 'gd_nesterov';
        end        
        
        options.max_iter = options_in.max_iter;
        options.w_init = options_in.w_init;
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
        elseif strcmp(method, 'gd_nesterov')
            options.step_alg = 'backtracking';
            options.step_init_alg = 'bb_init';
            [w_opt,~] = gd_nesterov(problem, options);            
        else 
            options.step_alg = 'backtracking';  
            options.mem_size = 5;
            [w_opt,~] = lbfgs(problem, options);              
        end
    end

end

