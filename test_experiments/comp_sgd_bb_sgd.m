function  comp_sgd_bb_sgd()

rng(1234);

    clc;
    clear;
    close all;

    %% Set algorithms
    %algorithms = {'BB-SGD-FIX','BB-SGD-BKT','SGD'};    
    %algorithms = {'SGD-0.1','SGD-0.01','BB-SGD-FIX','BB-SGD-BKT'}; 
    %algorithms = {'BB-SGD-BKT'};  
    %algorithms = {'BB-SGD-FIX','BB-SGD-BKT','BB-SGD-WOLFE', 'SGD-batch-1-step-0.01', 'SGD-batch-1-step-0.1', 'SGD-batch-10-step-0.01', 'SGD-batch-10-step-0.1'};  
    algorithms = {'BB-SGD-WOLFE','SGD-batch-1-step-0.1'};
    
    
    %% prepare dataset
    if 0
        % generate synthtic data        
        d = 100;
        n = 10000;
        data = logistic_regression_data_generator(n, d);
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        w_opt = data.w_opt;        
        lambda = 0.1;  
        
    elseif  1
        % IJCNN1
        [y_train,X_train] = libsvmread('../data/logistic_regression/ijcnn1/ijcnn1.tr'); 
        [y_test,X_test] = libsvmread('../data/logistic_regression/ijcnn1/ijcnn1.t');  
        
        X_train = featureNormalize(X_train);
        X_test = featureNormalize(X_test);       
        
        x_train = X_train';
        n_train = size(x_train, 2);
        x_train = [x_train; ones(1,n_train)];         
        y_train = y_train';    
        x_test = X_test';
        n_test = size(x_test, 2);        
        X_test = [x_test; ones(1,n_test)];          
        y_test = y_test'; 
        d = 22;
        d = d + 1;           
        lambda = 0;          
        
    elseif 0
        % load pre-created synthetic data        
        data = importdata('../data/logistic_regression/data_100d_10000.mat'); 
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        n = length(y_train);
        w_opt = data.w_opt;
        lambda = data.lambda;
        
    elseif 0
        % load real-world data
        data = importdata('../data/mushroom/mushroom.mat');
        x_in = data.X';
        y_in = data.y';    
        d = size(x_in,1);
        n = length(y_in);
        n_train = floor(n/2);        
        % split data into train and test data
        x_train = x_in(:,1:n_train);
        y_train = y_in(1:n_train);     
        x_test = x_in(:,n_train+1:end);
        y_test = y_in(n_train+1:end);          
        w_opt = zeros(d,1);        
        lambda = 0.1;
        
    elseif 1
        % generate synthtic data
        % sample data generating for training: y = w1*x1 + w2*x2 + ... * wd*1
        n = 10000;
        d = 50;
        std = 0.25;
        data = linear_regression_data_generator(n, d, std);
        
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;  
        
        % solution
        w_opt = pinv(x_train * x_train') * x_train * y_train';
        % for intersect
        d = d + 1;          
        lambda = 0.01;   
        
    else
        % load real-world data
        data = importdata('../data/linear_regression/Example.mat');
        x_in = data.X';
        y_in = data.Y';    
        d = size(x_in,1);
        n = length(y_in);  
        n_train = floor(n/2);        

        % scale features and set them to zero mean
        x_in = featureNormalize(x_in);   
        % add intercept term to x_in
        x_in = [x_in; ones(1,n)]; 
        
        x_train = x_in(:,1:n_train);
        y_train = y_in(1:n_train);     
        x_test = x_in(:,n_train+1:end);
        y_test = y_in(n_train+1:end);  
        
        % solution
        w_opt = pinv(x_train * x_train') * x_train * y_train';
        % for intersect
        d = d + 1;          
        lambda = 0.01;         

    end
    
    
    %% define problem definitions
    problem = logistic_regression(x_train, y_train, x_test, y_test, lambda);
    %problem = linear_regression(x_train, y_train, x_test, y_test, lambda);    
    
    
    %% calculate solution 
    %w_opt = problem.calc_solution(problem, 2000);    

    
    %% initialize
    w_init = randn(d,1);
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    
    
    %f_opt = problem.cost(w_opt); 
    %fprintf('f_opt: %.24e\n', f_opt); 
    %f_opt =  2.159729221311369462554808e-01;
    f_opt =  1.9629817795736132e-01;
     

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_optgap = 10^-36;
        options.max_epoch = 100;
        options.verbose = true;
        options.lambda = lambda;
        options.permute_on = 1; 
        options.f_opt = f_opt;
        
        
        switch algorithms{alg_idx}
            
            case {'SGD-batch-1-step-0.01'}
                
                options.batch_size = 1;
                options.step_init = 0.01;
                options.step_alg = 'decay';
                options.lambda = 0.001;

                [w_list{alg_idx}, info_list{alg_idx}] = sgd_tmp(problem, options);  
                
            case {'SGD-batch-1-step-0.1'}
                
                options.batch_size = 1;
                options.step_init = 0.1;
                options.step_alg = 'decay';
                options.lambda = 0.001;

                [w_list{alg_idx}, info_list{alg_idx}] = sgd_tmp(problem, options);  
                
            case {'SGD-batch-10-step-0.01'}
                
                options.batch_size = 10;
                options.step_init = 0.01;
                options.step_alg = 'decay';
                options.lambda = 0.001;

                [w_list{alg_idx}, info_list{alg_idx}] = sgd_tmp(problem, options);  
                
            case {'SGD-batch-10-step-0.1'}
                
                options.batch_size = 10;
                options.step_init = 0.1;
                options.step_alg = 'decay';
                options.lambda = 0.001;

                [w_list{alg_idx}, info_list{alg_idx}] = sgd_tmp(problem, options);                   
                
            case {'BB-SGD-FIX'}
                
                options.batch_size = 50;
                options.step = 1;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = bb_sgd(problem, options);   
                
            case {'BB-SGD-BKT'}
                
                options.batch_size = 50;
                options.step = 1;
                options.step_alg = 'backtracking';

                [w_list{alg_idx}, info_list{alg_idx}] = bb_sgd(problem, options);   
                
            case {'BB-SGD-WOLFE'}
                
                options.batch_size = 50;
                options.step = 1;
                options.step_alg = 'strong_wolfe';

                [w_list{alg_idx}, info_list{alg_idx}] = bb_sgd(problem, options);                   

            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
        
    end
    
    fprintf('\n\n');
    
    
    %% plot all
    close all;
    % display cost vs grads
    display_graph('numofgrad','cost', algorithms, w_list, info_list);
    % display optimality gap vs grads
    if options.f_opt ~= -Inf
        display_graph('numofgrad','optimality_gap', algorithms, w_list, info_list);
    end
    % display cost vs grads
    display_graph('numofgrad','gnorm', algorithms, w_list, info_list);  
    if options.f_opt ~= -Inf    
        display_graph('time','optimality_gap', algorithms, w_list, info_list); 
    end  
    

    bb_cnt = 0;
    bb_algorithms = cell(1);    
    bb_w_list = cell(1);
    bb_info_list = cell(1);    
    for i=1:length(algorithms)
        if strfind(algorithms{i}, 'BB')
            bb_cnt = bb_cnt + 1;
            bb_algorithms{bb_cnt} = algorithms{i};
            bb_w_list{bb_cnt} = w_list{i};
            bb_info_list{bb_cnt} = info_list{i};
        end
    end

    display_graph('numofgrad','K', bb_algorithms, bb_w_list, bb_info_list);     
    display_graph('iter','K', bb_algorithms, bb_w_list, bb_info_list);        

    
end




