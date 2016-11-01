function  demo()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 24, 2016


    clc;
    clear;
    close all;

    %% generate synthtic data        
    % set number of dimensions
    d = 3;
    % set number of samples    
    n = 300;
    % generate data
    data = logistic_regression_data_generator(n, d);
    % set train data
    x_train = data.x_train;
    y_train = data.y_train;  
    % set test data
    x_test = data.x_test;
    y_test = data.y_test;               
    % set lambda 
    lambda = 0.1;
        
    
    %% define problem definitions
    problem = logistic_regression(x_train, y_train, x_test, y_test, lambda); 
    
    
    %% calculate solution 
    w_star = problem.calc_solution(problem, 1000, 0.01);
     

    %% general options for optimization algorithms   
    % generate initial point
    options.w_init = randn(d,1);
    % set iteration optimality gap tolerance
    options.tol_optgap = -Inf;
    % set max epoch
    options.max_epoch = 100;
    % set verbose mode
    options.verbose = true;
    % set regularization parameter    
    options.lambda = lambda;
    % set solution
    options.f_sol = problem.cost(w_star);
    % set batch sizse    
    options.batch_size = 10;
    % set stepsize algorithm and stepsize
    options.step_alg = 'fix';
    options.step = 0.01;     

    
    %% perform algorithms SGD
    [w_sgd, info_list_sgd] = sgd(problem, options);  
    % predict
    y_pred_sgd = problem.prediction(w_sgd);
    % calculate accuracy
    accuracy_sgd = problem.accuracy(y_pred_sgd); 
    fprintf('Classificaiton accuracy: %s: %.4f\n', 'SGD', accuracy_sgd);
    % convert from {1,-1} to {1,2}
    y_pred_sgd(y_pred_sgd==-1) = 2;
    y_pred_sgd(y_pred_sgd==1) = 1;
        
    
    %% perform algorithms SVRG
    [w_svrg, info_list_svrg] = svrg(problem, options);  
    % predict    
    y_pred_svrg = problem.prediction(w_svrg);
    % calculate accuracy
    accuracy_svrg = problem.accuracy(y_pred_svrg); 
    fprintf('Classificaiton accuracy: %s: %.4f\n', 'SVRG', accuracy_svrg);
    % convert from {1,-1} to {1,2}
    y_pred_svrg(y_pred_svrg==-1) = 2;
    y_pred_svrg(y_pred_svrg==1) = 1;
                
    
    %% plot all
    % display cost vs grads
    display_graph('grad_calc_count','cost', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_list_sgd, info_list_svrg});
    % display optimality gap vs grads
    display_graph('grad_calc_count','optimality_gap', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_list_sgd, info_list_svrg});
    % convert from {1,-1} to {1,2}
    y_train(y_train==-1) = 2;
    y_train(y_train==1) = 1;
    y_test(y_test==-1) = 2;
    y_test(y_test==1) = 1;  
    % display classification results    
    display_classification_result(problem, {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {y_pred_sgd, y_pred_svrg}, {accuracy_sgd, accuracy_svrg}, x_train, y_train, x_test, y_test);    
    
end


