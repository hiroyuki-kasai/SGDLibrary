function  demo()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 24, 2016
% Modified by H.Kasai on Nov. 03, 2016


    clc;
    clear;
    close all;

    %% generate synthetic data        
    % set number of dimensions
    d = 3;
    % set number of samples    
    n = 300;
    % generate data
    data = logistic_regression_data_generator(n, d);
        
    
    %% define problem definitions
    problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test); 
    
    
    %% perform algorithms SGD and SVRG 
    options.w_init = data.w_init;    
    options.step_init = 0.01;       
    [w_sgd, info_sgd] = sgd(problem, options);  
    [w_svrg, info_svrg] = svrg(problem, options);
    
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('grad_calc_count','cost', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});

end


