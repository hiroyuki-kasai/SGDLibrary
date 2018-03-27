function  demo_ext()
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
    d = 2;
    % set number of samples    
    n = 300;
    % generate data
    data = logistic_regression_data_generator(n, d);
        
    
    %% define problem definitions
    problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test); 
    
    
    %% calculate optimal solution for optimality gap
    w_opt = problem.calc_solution(1000);
    options.f_opt = problem.cost(w_opt);

    
    %% set options for convergence animation
    options.max_epoch = 100;    
    options.store_w = true;

    
    %% perform algorithms SGD and SVRG 
    options.w_init = data.w_init;
    options.step_init = 0.01;       
    [w_sgd, info_sgd] = sgd(problem, options);  
    [w_svrg, info_svrg] = svrg(problem, options);
    
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('grad_calc_count','cost', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});
    display_graph('grad_calc_count','optimality_gap', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});    
    
    
    %% calculate classification accuracy
    % for SGD
    % predict
    y_pred_sgd = problem.prediction(w_sgd);
    % calculate accuracy
    accuracy_sgd = problem.accuracy(y_pred_sgd); 
    fprintf('Classificaiton accuracy: %s: %.4f\n', 'SGD', accuracy_sgd);
    % convert from {1,-1} to {1,2}
    y_pred_sgd(y_pred_sgd==-1) = 2;
    y_pred_sgd(y_pred_sgd==1) = 1; 
    
    % for SVRG
    % predict    
    y_pred_svrg = problem.prediction(w_svrg);
    % calculate accuracy
    accuracy_svrg = problem.accuracy(y_pred_svrg); 
    fprintf('Classificaiton accuracy: %s: %.4f\n', 'SVRG', accuracy_svrg);
    % convert from {1,-1} to {1,2}
    y_pred_svrg(y_pred_svrg==-1) = 2;
    y_pred_svrg(y_pred_svrg==1) = 1;
                
    
    %% display classification results 
    % convert from {1,-1} to {1,2}
    data.y_train(data.y_train==-1) = 2;
    data.y_train(data.y_train==1) = 1;
    data.y_test(data.y_test==-1) = 2;
    data.y_test(data.y_test==1) = 1;  
    % display results
    display_classification_result(problem, {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {y_pred_sgd, y_pred_svrg}, {accuracy_sgd, accuracy_svrg}, data.x_train, data.y_train, data.x_test, data.y_test);    
    
    
    %% display convergence animation
    draw_convergence_animation(problem, {'SGD', 'SVRG'}, {info_sgd.w, info_svrg.w}, options.max_epoch, 0.1);        
    
end


