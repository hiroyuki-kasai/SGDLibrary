function  comp_svrg_svrg_bb()

    clc;
    clear;
    close all;

    %% Set algorithms
    algorithms = {'SVRG-0.1','SVRG-BB-0.1','SVRG-0.01','SVRG-BB-0.01','SVRG-0.001','SVRG-BB-0.001','SVRG-0.0001','SVRG-BB-0.0001'};     
    %algorithms = {'SVRG-0.0001','SVRG-BB-0.0001'};  
    
    
    %% prepare dataset
    if 0
        % generate synthtic data        
        d = 30;
        n = 10000;
        data = logistic_regression_data_generator(n, d);
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        w_star = data.w_star;        
        lambda = 0.1;        
        
    elseif 0
        % load pre-created synthetic data        
        data = importdata('../data/logistic_regression/data_100d_10000.mat'); 
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        n = length(y_train);
        w_star = data.w_star;
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
        w_star = zeros(d,1);        
        lambda = 0.1;
        
    elseif 0
        % generate synthtic data
        % sample data generating for training: y = w1*x1 + w2*x2 + ... * wd*1
        n = 1000;
        d = 5;
        std = 0.25;
        data = linear_regression_data_generator(n, d, std);
        
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;  
        
        % solution
        w_star = pinv(x_train * x_train') * x_train * y_train';
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
        w_star = pinv(x_train * x_train') * x_train * y_train';
        % for intersect
        d = d + 1;          
        lambda = 0.01;         

    end
    
    % set plot_flag
    if d > 4
        plot_flag = false;  % too high dimension  
    else
        plot_flag = true;
    end    

    
    %% define problem definitions
    problem = logistic_regression(x_train, y_train, x_test, y_test, lambda);
    
    
    %% calculate solution 
    w_star = problem.calc_solution(problem, 500);    

    
    %% initialize
    w_init = randn(d,1);
    batch_size = 10;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    
    
    %% calculate solution
    if norm(w_star)
    else
        % calculate solution
        w_star = problem.calc_solution(problem, 1000);
    end
    f_opt = problem.cost(w_star); 
    fprintf('f_opt: %.24e\n', f_opt);    
     

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
            
            case {'SVRG-0.1'}
                
                options.batch_size = batch_size;
                options.step = 0.1;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);    
                
            case {'SVRG-BB-0.1'}
                
                options.batch_size = batch_size;
                options.step = 0.1;

                [w_list{alg_idx}, info_list{alg_idx}] = svrg_bb(problem, options);                     

            case {'SVRG-0.01'}
                
                options.batch_size = batch_size;
                options.step = 0.01;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);    
                
            case {'SVRG-BB-0.01'}
                
                options.batch_size = batch_size;
                options.step = 0.01;

                [w_list{alg_idx}, info_list{alg_idx}] = svrg_bb(problem, options);   
                

            case {'SVRG-0.001'}
                
                options.batch_size = batch_size;
                options.step = 0.001;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);    
                
            case {'SVRG-BB-0.001'}
                
                options.batch_size = batch_size;
                options.step = 0.001;

                [w_list{alg_idx}, info_list{alg_idx}] = svrg_bb(problem, options);   
                
            case {'SVRG-0.0001'}
                
                options.batch_size = batch_size;
                options.step = 0.0001;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);    
                
            case {'SVRG-BB-0.0001'}
                
                options.batch_size = batch_size;
                options.step = 0.0001;

                [w_list{alg_idx}, info_list{alg_idx}] = svrg_bb(problem, options);                    
                
                 

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
    display_graph_two_comparison('cost', algorithms, w_list, info_list);
    % display optimality gap vs grads
    if options.f_opt ~= -Inf
        display_graph_two_comparison('optimality_gap', algorithms, w_list, info_list);
    end
    
end




