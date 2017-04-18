function [] = test_lasso()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'APG-BKT', 'APG-TFOCS-BKT'};     
    end    
    
    
    %% prepare dataset
    if 1
        % generate synthtic data        
        n = 1e4; 
        d = 100; 
        A = randn(d,n); 
        b = randn(d,1); 
        lambda = 5;
    else
    end
    
    
    %% define problem definitions
    problem = lasso(A, b, lambda);

    
    %% initialize
    w_init = rand(n,1); 
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
                [w_list{alg_idx}, info_list{alg_idx}] = gd(problem, options);
                
            case {'PG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = gd(problem, options);     
                
            case {'APG-BKT'}
                
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = gd_nesterov(problem, options);
                
            case {'APG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = gd_nesterov(problem, options);  
                
            case {'L-BFGS-BKT'}
                
                options.step_alg = 'backtracking';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);
                
            case {'L-BFGS-WOLFE'}
                
                options.step_alg = 'strong_wolfe';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);               
                
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




