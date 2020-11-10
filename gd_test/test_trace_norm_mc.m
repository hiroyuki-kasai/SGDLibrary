function [] = test_trace_norm_mc()

    clc;
    clear;
    close all;
    
    rng('default');
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        %algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'APG-BKT', 'APG-TFOCS-BKT', 'L-BFGS-BKT'};      
        algorithms = {'PG-TFOCS-BKT', 'APG-TFOCS-BKT', 'L-BFGS-BKT'}; 
    end    
    
    
    %% prepare dataset
    if 1
        % generate synthtic data        
        n = 100; 
        m = 50; 
        r = 10; 
        density = 0.2; 
        lambda = 5;
        M = randn(m,r)*randn(r,n); 
        mask = (rand(m,n)<density);
    else
    end
    
    
    %% define problem definitions
    problem = trace_norm_matrix_completion(M, mask, lambda);

    
    %% initialize
    w_init = randn(m*n, 1);
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 100;
        options.verbose = true;  
        options.f_opt = 0;
        options.store_w = false;

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
    
    % display iter vs. cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display iter vs. trace (nuclear) norm
    display_graph('iter','reg', algorithms, w_list, info_list); 
end

