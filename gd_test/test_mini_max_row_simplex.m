function [] = test_mini_max_row_simplex()

    % min f(w) = max a_i^T x : x in probability simplex
    % where a_1^T, a_2^T, ..., a_n^T are the rows of A

    close all
    clear 
    clc

    %randn('seed',315);

    %% generate dataset
    n = 80;
    p = 50;
    A = randn(n, p);

    %% define problem definitions
    problem = mini_max_row_simplex(A);


    %% set initial solution
    x_init = 1/p * ones(p, 1); 
    
    
    %% Perform algorithms
    clear options;
    options.w_init = x_init;
    options.tol_gnorm = -inf;
    options.tol_optgap = -inf;
    options.max_epoch = 500;
    options.verbose = true; 
    options.store_w = true;

    options.step_alg = 'decay-7';
    
    options.step_init = 1;    
    algorithms{1} = 'Proximal Subgradient (step: 1)';      
    [w_list{1}, info_list{1}] = subg(problem, options); 

    options.step_init = 0.2;
    algorithms{2} = 'Proximal Subgradient (step: 0.2)';     
    [w_list{2}, info_list{2}] = subg(problem, options); 


    %% plot all
    % display iter vs. cost
    display_graph('iter', 'cost', algorithms, w_list, info_list);
 
end