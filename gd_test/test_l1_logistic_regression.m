function [] = test_l1_logistic_regression()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        %algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'APG-BKT', 'APG-TFOCS-BKT', 'Newton-CHOLESKY', 'NCG-BKT','L-BFGS-TFOCS'};
        algorithms = {'APG-BKT', 'APG-TFOCS-BKT'};
    end    
    
    
    %% prepare dataset
    if 1
        % generate synthtic data        
        d = 100;
        n = 1000;
        data = logistic_regression_data_generator(n, d);
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        w_opt = data.w_opt;        
        lambda = 0.1;   
    else
        % load pre-created synthetic data        
        data = importdata('../data/logistic_regression/data_100d_10000.mat'); 
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;          
        d = size(x_train,1);
        n = length(y_train);
        w_opt = data.w_star;
        lambda = data.lambda;        
    end
    
    %% define problem definitions
    problem = l1_logistic_regression(x_train, y_train, x_test, y_test, lambda);

    
    %% calculate solution
    if norm(w_opt)
    else
        % calculate solution
        w_opt = problem.calc_solution(1000, 0.05);
    end
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);   
    
    
    %% initialize
    w_init = rand(d,1); 
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 300;
        options.verbose = true;  

        switch algorithms{alg_idx}
            case {'PG-BKT'}
                
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);
                
            case {'PG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);     
                
            case {'APG-BKT'}
                
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = ag(problem, options);
                
            case {'APG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = ag(problem, options);  
                
            case {'L-BFGS-BKT'}
                
                options.step_alg = 'backtracking';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);
                
            case {'L-BFGS-WOLFE'}
                
                options.step_alg = 'strong_wolfe';   
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);  
 
            case {'L-BFGS-TFOCS'}
                
                options.step_alg = 'tfocs_backtracking';  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);  

            case {'BFGS-TFOCS'}
                
                options.step_alg = 'tfocs_backtracking'; 
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);  
                
            case {'Newton-CHOLESKY'}

                options.sub_mode = 'CHOLESKY';                
                options.step_alg = 'backtracking';
                %options.step_alg = 'tfocs_backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);

            case {'NCG-BKT'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'backtracking'; 
                %options.step_alg = 'tfocs_backtracking';
                %options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);                
                
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
    
    % display iter vs cost/gnorm
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display iter vs. l1 norm, i.e. the toral number of non-zero elements 
    display_graph('iter','reg', algorithms, w_list, info_list); 
    
end




