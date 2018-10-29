function demo()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 24, 2016
% Modified by H.Kasai on Nov. 03, 2016

    clc; clear; close all;
    run_me_first;
    %% generate synthetic data        
    % set number of dimensions
    d = 100;
    % set number of samples    
    n = 65535;
    % generate data
    data = logistic_regression_data_generator(n, d);

    %% define problem definitions
    problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test); 
    
    %% perform algorithms SGD and SVRG 
    options.w_init = data.w_init;    
    options.step_init = 0.01; 
    options.verbose = 2;
    options.max_epoch = 25;
    options.max_iter = 25;
    %options.sub_mode = 'STANDARD';
    %[w_sgd, info_sgd] = sgd(problem, options);  
    %[w_svrg, info_svrg] = svrg(problem, options);
    
    %[w_nt, info_nt] = newton(problem, options);
    options.f_opt = 0;
    
    options.subsamp_hess_size = n;
    options.sub_mode = 'Uniform';
    [w_nt, info_nt] = subsamp_newton(problem, options);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    options.subsamp_hess_size = 4*d;
    options.sub_mode = 'Uniform';
    [w_uni, info_uni] = subsamp_newton(problem, options);   
    options.sub_mode = 'LS';
    [w_ls, info_ls] = subsamp_newton(problem, options);
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('iter','cost', {'Newton', 'Uniform', 'Leverage scroe'},...
                  {w_nt, w_uni, w_ls}, {info_nt, info_uni, info_ls});
              
%     display_graph('iter','optimality_gap', {'SGD', 'SVRG', 'SSN'},...
%                   {w_sgd, w_svrg, w_ssn}, {info_sgd, info_svrg, info_ssn});
    
end
