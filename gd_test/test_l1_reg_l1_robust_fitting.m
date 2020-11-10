function [] = test_l1_reg_l1_robust_fitting()

    % f(x) = || A * w - b ||_1 + 2 * || w ||_1
    
    close all;
    clear;
    clc;
    
    %% generate dataset    
    if 0
        n = 4;
        A = [0.6324 0.9575 0.9572 0.4218;
            0.0975 0.9649 0.4854 0.9157;
            0.2785 0.1576 0.8003 0.7922; 
            0.5469 0.9706 0.1419 0.9595];
        b = [0.6843;0.6706; 0.4328; 0.8038];
        w_opt = pinv(A) * b;
        lambda = 2;
        step_init = 1;
    else
        n = 500; 
        d = 200; 
        A = randn(d,n); 
        w_opt = randn(n,1); 
        noise = 0.;
        b = A * w_opt + noise * randn(d, 1);
        lambda = 2;    
        step_init = 0.1;        
    end
    
    max_iter = 100;
    w_init = zeros(n,1);

    %% (1) Solve as f(x) = || A * w - b ||_1 + lambda_c || w ||_1
    % Two terms are handled by subgradient, i.e., 
    % Subgradient descent algorithm
    
    % define problem definitions
    problem = l1_robust_fitting(A, b, lambda, 0);  
    f_opt = problem.cost(w_opt) - lambda * norm(w_opt);
    
    algorithms{1} = 'Subgradient (decay)'; 
    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = max_iter;
    options.f_opt = f_opt;     
    options.verbose = true; 
    options.store_w = true;
    options.step_init = step_init;
    options.step_alg = 'decay-7';
    [w_list{1}, info_list{1}] = subgrad(problem, options);
    
    algorithms{2} = 'Subgradient (adaptive decay)'; 
    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = max_iter;
    options.f_opt = f_opt;       
    options.verbose = true; 
    options.store_w = true;
    options.linesearchfun = @my_stepalg;  % set my_stepalg (user-defined stepsize algorithm) 
    [w_list{2}, info_list{2}] = subgrad(problem, options);    
    
    
    
    %% (2) Solve as f(x) = || A * w - b ||_1 + lambda_r || w ||_1
    % The first term is handled by subgradient, and the second one is by proximal operator
    % Proximal Subgradient descent algorithm    

    % define problem definitions
    problem = l1_robust_fitting(A, b, 0, lambda);    

    algorithms{3} = 'Proximal Subgradient (decay)';     
    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = max_iter;
    options.f_opt = f_opt;       
    options.verbose = true; 
    options.store_w = true;
    options.step_init = step_init;
    options.step_alg = 'decay-7';
    [w_list{3}, info_list{3}] = subgrad(problem, options);    
    
    algorithms{4} = 'Proximal Subgradient (adaptive decay)';     
    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = max_iter;
    options.f_opt = f_opt;       
    options.verbose = true; 
    options.store_w = true;
    options.linesearchfun = @my_stepalg;  % set my_stepalg (user-defined stepsize algorithm) 
    [w_list{4}, info_list{4}] = subgrad(problem, options);        
    
  

    %% plot all
    % display iter vs. cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display time vs. optimality_gap
    display_graph('iter','optimality_gap', algorithms, w_list, info_list); 
    %
    %display_graph('iter','subgnorm', algorithms, w_list, info_list);      
  
end


%% define user-defined stepsize algorithm
function [step, nothing] = my_stepalg(step_alg, problem, w, w_old, grad, grad_old, prev_step, options)
    iter = options.iter;
    subgrad = options.subgrad;
    step = sqrt(2)/( norm(subgrad) * sqrt(iter+2));
    nothing = [];
end  

