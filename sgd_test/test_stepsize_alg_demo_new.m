function test_stepsize_alg_demo()
% demonstration file for original stepsize algorithm.
%
% This file illustrates how to set user's own stepsize algorithm in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Sep. 25, 2017


    clc;
    clear;
    close all;

    %% generate synthetic data        
    % set number of dimensions
    d = 10;
    % set number of samples    
    n = 1000;
    % generate data
    data = logistic_regression_data_generator(n, d);
        
    
    %% define problem definitions
    problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test); 
    
    
%     %% define user-defined stepsize algorithm
%     function step = my_stepalg(iter, options)
%         step = options.step_init / (10 + iter*0.5);
%     end       
%     
    
    %% perform algorithms SGD and SVRG 
    options.w_init = data.w_init;    
    options.step_init = 0.01;  
    options.verbose = 2;
    
    options.step_alg = 'fix';
    [w_sgd_fix, info_sgd_fix] = sgd(problem, options); 
    
    options.step_alg = 'decay';
    [w_sgd_decay, info_sgd_decay] = sgd(problem, options);      
    
    options.step_alg = 'decay-2';
    [w_sgd_decay2, info_sgd_decay2] = sgd(problem, options);       
    
    options.stepsizefun = @my_stepalg;  % set my_stepalg (user-defined stepsize algorithm)
    [w_sgd_my, info_sgd_my] = sgd(problem, options);      
    
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('grad_calc_count','cost', {'SGD (fix)','SGD (decay)', 'SGD (decay-2)', 'SGD (My stepsize algorithm)'}, ...
            {w_sgd_fix, w_sgd_decay w_sgd_decay2, w_sgd_my}, {info_sgd_fix, info_sgd_decay, info_sgd_decay2, info_sgd_my});

end

    %% define user-defined stepsize algorithm
    function step = my_stepalg(iter, options)
        step = options.step_init / (10 + iter*0.5);
    end       
    


