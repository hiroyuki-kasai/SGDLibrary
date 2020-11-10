function [] = test_l1_robust_fitting()

    % f(x) = || A * w - b ||_1
    
    close all;
    clear;
    clc;
    

     %% prepare dataset
    n = 100; 
    d = 500; 
    A = randn(d,n); 
    b = randn(d,1); 
    x_init = zeros(n,1);
    

    %% (1) Solve as f(x) = || A * w - b ||_1
    % Two terms are handled by subgradient, i.e., 
    % Subgradient descent algorithm
    
    % define problem definitions
    problem = l1_robust_fitting(A, b, 0, 0); 
    
    
    %% Calculate optimal value
    fprintf('Calculating the optimal value ...');
    clear options;
    options.w_init = x_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = 10000;
    options.step_init = 0.01;
    options.step_alg = 'decay-2';
    [w_opt, ~] = subgrad(problem, options);  
    f_opt = problem.cost(w_opt);
    fprintf('done.\n');
    
    clear options;
    options.w_init = x_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = 5000;
    options.f_opt = f_opt;
    options.verbose = true; 
    options.step_init = 0.01;
    
    algorithms{1} = 'Subgradient (t_k=0.01/sqrt(k+1))';     
    options.step_alg = 'decay-5';
    [w_list{1}, info_list{1}] = subgrad(problem, options);
    
    algorithms{2} = 'Subgradient (t_k=0.01/(k+1))';     
    options.step_alg = 'decay-2';
    [w_list{2}, info_list{2}] = subgrad(problem, options);    
   
  

    %% plot all
    % display iter vs. cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display optimality_gap vs. cost
    display_graph('iter','optimality_gap', algorithms, w_list, info_list);    
   
  
end

