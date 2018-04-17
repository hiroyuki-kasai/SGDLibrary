function comp_inexact_hessian()
% demonstration file for SGDLibrary.
%
% This file illustrates test comparisons between various inexact
% gradient/Hessian algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on April 16, 2018


    clc;
    clear;
    close all;
    
    %rng('default');


    %% load dataset(a9a) 

    load('../data/libsvm/a9a_new.mat'); %load X, Y, n ,d, w
    fprintf('n = %d, d = %d\n',n,d);
    
    data.x_train = X';
    data.y_train = Y';
    data.x_test = [];
    data.y_test = [];    

 
    
    %% define problem definitions
    alpha = 1e-3;
    problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test, alpha); 
    %problem = linear_regression(data.x_train, data.y_train, data.x_test, data.y_test, alpha);
    
    
    % generate initial value
    w_init = rand(d,1);
    max_epoch = 20;
    
    
    
    %% calcualte f_opt
    options.max_iter = max_epoch;
    options.f_opt = -Inf;
    options.w_init = w_init;  
    options.verbose = 2;
    options.sampling_scheme = 'adaptive'; % 'exponential', 'linear'
    [w_scr_opt, ~] = scr(problem, options);     
    f_opt = problem.cost(w_scr_opt);
    options.f_opt = f_opt; 
    

    
    
    %% perform solvers
    
    % SGD
    options.step_init = 0.1; 
    options.max_epoch = max_epoch;    
    [w_sgd, info_sgd] = sgd(problem, options); 
    
    
    % SVRG
    options.step_init = 0.1;
    options.max_epoch = max_epoch/2;
    [w_svrg, info_svrg] = svrg(problem, options);  
    
    
    % L-BFGS
    options.step_init = 0.1;
    options.max_iter = max_epoch;
    options.step_alg = 'strong_wolfe';  
    options.mem_size = 20;    
    [w_lbfgs, info_lbfgs] = lbfgs(problem, options);
    
    
%     % Sub-sampled Newton with uniform sampling
%     options.sub_mode = 'Uniform'; 
%     options.subsamp_hess_size = 100*d;
%     [w_sub_newton, info_sub_newton] = subsamp_newton(problem, options); 
%
%     % Sub-sampled Newton with row norm squares
%     options.sub_mode = 'RNS'; 
%     options.subsamp_hess_size = 20*d;
%     [w_sub_newton, info_sub_newton] = subsamp_newton(problem, options);   

    % Sub-sampled Newton with leverage scores
    options.sub_mode = 'LS'; 
    options.subsamp_hess_size = 20*d;
    options.hess_update_freq = 10;
    [w_sub_newton, info_sub_newton] = subsamp_newton(problem, options);         
    
    
    % TR
    clear options;
    options.max_iter = max_epoch;
    options.f_opt = f_opt;
    options.w_init = w_init;  
    options.verbose = 2;  
    options.gradient_sampling = 0;
    options.Hessian_sampling = 0;     
    options.subproblem_solver = 'GLTR'; %'cg';
    options.sampling_scheme = 'none';
    [w_tr, info_tr] = subsamp_tr(problem, options);    
    

    % Sub-sampled TR with adaptive Hessian sampling (without gradient sampling)
    clear options;
    options.max_iter = max_epoch;
    options.f_opt = f_opt;
    options.w_init = w_init;  
    options.verbose = 2;    
    options.subproblem_solver = 'GLTR'; %'cg';
    options.sampling_scheme = 'exponential';
    [w_tr_ad, info_tr_ad] = subsamp_tr(problem, options);

 
    % ARC
    clear options;
    options.max_iter = max_epoch;
    options.f_opt = f_opt;
    options.w_init = w_init;  
    options.verbose = 2;  
    options.gradient_sampling = 0;
    options.Hessian_sampling = 0;    
    options.sampling_scheme = 'none';    
    options.subproblem_solver = 'lanczos';    
    [w_arc, info_arc] = scr(problem, options);  
    
    
    % SCR with adaptive Hessian sampling (without gradient sampling)
    clear options;
    options.max_iter = max_epoch;
    options.f_opt = f_opt;
    options.w_init = w_init;  
    options.verbose = 2;  
    options.sampling_scheme = 'adaptive';    
    options.subproblem_solver = 'lanczos';    
    [w_scr_ad, info_scr_ad] = scr(problem, options);  

    
    
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('grad_calc_count','optimality_gap', {'SGD', 'SVRG', 'LBFGS', 'Subsampled Newton', 'TR', 'Subsampled TR (adaptive)', 'ARC', 'Subsampled CR (adaptive)'}, ...
        {w_sgd, w_svrg, w_lbfgs, w_sub_newton, w_tr, w_tr_ad, w_arc, w_scr_ad}, {info_sgd, info_svrg, info_lbfgs, info_sub_newton, info_tr, info_tr_ad, info_arc, info_scr_ad}); 
    

    %% display cost/optimality gap vs time
    display_graph('time','optimality_gap', {'SGD', 'SVRG', 'LBFGS', 'Subsampled Newton', 'TR', 'Subsampled TR (adaptive)', 'ARC', 'Subsampled CR (adaptive)'}, ...
        {w_sgd, w_svrg, w_lbfgs, w_sub_newton, w_tr, w_tr_ad, w_arc, w_scr_ad}, {info_sgd, info_svrg, info_lbfgs, info_sub_newton, info_tr, info_tr_ad, info_arc, info_scr_ad});    

end


