function demo_paper()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD, SVRG, SQN, and SVRG-LBFGS algorithms.
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
    
    w_opt = problem.calc_solution(1000);
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);      
    
    
    %% perform algorithms SGD and SVRG 
    options.w_init = data.w_init;    
    options.step_init = 0.01; 
    options.verbose = 2;
    options.f_opt = f_opt;
    [w_sgd, info_sgd] = sgd(problem, options);  
    [w_svrg, info_svrg] = svrg(problem, options);
    
    batch_size = 10;
    options.batch_size = batch_size;
    options.batch_hess_size = batch_size * 20;        
    [w_sqn, info_sqn] = sag(problem, options);     
    
    batch_size = 10;
    options.batch_size = batch_size;
    options.batch_hess_size = batch_size * 20;        
    options.step_init = 0.01 * options.batch_size;
    options.step_alg = 'fix';
    options.sub_mode = 'SVRG-LBFGS';
    options.mem_size = 20;

    [w_svrg_lbfgs, info_svrg_lbfgs] = slbfgs(problem, options);       
    
    algorithms = {'SGD', 'SVRG', 'SQN', 'SVRG-LBFGS'};
    w_list = {w_sgd, w_svrg, w_sqn, w_svrg_lbfgs};
    info_list = {info_sgd, info_svrg, info_sqn, info_svrg_lbfgs};
    
    
    %% display cost/optimality gap vs number of gradient evaluations
    display_graph('grad_calc_count','cost', algorithms, w_list, info_list);
    display_graph('grad_calc_count','optimality_gap', algorithms, w_list, info_list);    
    
    % display classification results
    y_pred_list = cell(length(algorithms),1);
    accuracy_list = cell(length(algorithms),1);    
    for alg_idx=1:length(algorithms)  
        if ~isempty(w_list{alg_idx})           
            p = problem.prediction(w_list{alg_idx});
            % calculate accuracy
            accuracy_list{alg_idx} = problem.accuracy(p); 

            fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, problem.accuracy(p));

            % convert from {1,-1} to {1,2}
            p(p==-1) = 2;
            p(p==1) = 1;
            % predict class
            y_pred_list{alg_idx} = p;
        else
            fprintf('Classificaiton accuracy: %s: Not supported\n', algorithms{alg_idx});
        end
    end 

    % convert from {1,-1} to {1,2}
    data.y_train(data.y_train==-1) = 2;
    data.y_train(data.y_train==1) = 1;
    data.y_test(data.y_test==-1) = 2;
    data.y_test(data.y_test==1) = 1;  
    %if plot_flag        
        display_classification_result(problem, algorithms, w_list, y_pred_list, accuracy_list, data.x_train, data.y_train, data.x_test, data.y_test);    
    %end    

end


