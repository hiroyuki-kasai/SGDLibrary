function [] = test_lasso()

    %clc;
    clear;
    close all;
    
    rng('default')
    
     
    %% Set algorithms
    %algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'PG-FIX', 'SUBG-DEC', 'SUBG-BKT', 'APG-TFOCS-BKT', 'FISTA', 'CD-LASSO', 'ADMM-LASSO'};
    algorithms = {'SMOOTH-FIX', 'SMOOTH-BKT', 'PG-WOLFE', 'PG-TFOCS-BKT', 'PG-FIX', 'SUBG-DEC', 'SUBG-BKT', 'APG-BKT', 'APG-TFOCS-BKT', 'FISTA', 'ADMM-LASSO'};
    %algorithms = {'SUBG-DEC', 'SUBG-BKT', 'SUBG-TFOCS-BKT'};
    %algorithms = {'SUBG-DEC', 'SUBG-BKT', 'SUBG-TFOCS-BKT', 'SMOOTH-FIX', 'FISTA', 'ADMM-LASSO'};
    algorithms = {'SUBG-DEC', 'SUBG-BKT'};
    
    
    %% prepare dataset
    if 1   
        n = 1280; 
        d = 100;         
        k = 15;                                     % cardinality of nonzero elements
        [A,~] = qr(randn(n,d),0);                   
        A = A';                                    
        p = randperm(n); 
        p = p(1:k);                                 % select location of k nonzeros
        x0 = zeros(n,1); 
        x0(p) = randn(k,1);                         
        b = A*x0 + .02*randn(d, 1);                 % add random noise   
        lambda_max = norm( A'*b, 'inf' );
        lambda = 0.1*lambda_max;
    else          
        n = 500; 
        d = 100; 
        A = randn(d,n); 
        b = randn(d,1); 
        lambda = 5;
    end
    
    
    
    %% initialize
    w_init = rand(n,1); 
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    

    %% perform algorithms
    for alg_idx=1:length(algorithms)
        
        %% define problem definitions
        clear problem;
        problem = lasso(A, b, lambda, 'prox_reg');
    
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_epoch = 100;
        options.max_iter = options.max_epoch;
        options.verbose = true; 
        options.f_opt = 0;
        options.store_w = false;

        switch algorithms{alg_idx}
            case {'SMOOTH-FIX'}
                smooth_mu = 0.1;
                clear problem;
                problem = lasso(A, b, lambda, 'smooth', smooth_mu);
                
                options.step_alg = 'fix';
                options.step_init = 1/(problem.L+lambda/smooth_mu); 
                [w_list{alg_idx}, info_list{alg_idx}] = smoothing_gd(problem, options);
                
            case {'SMOOTH-BKT'}
                smooth_mu = 0.1;
                clear problem;
                problem = lasso(A, b, lambda, 'smooth', smooth_mu);
                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = smoothing_gd(problem, options);                
                
            case {'SUBG-DEC'}
                
                options.step_alg = 'decay-7';
                options.step_init = 1/problem.L; 
                [w_list{alg_idx}, info_list{alg_idx}] = subg(problem, options);
                
            case {'SUBG-BKT'}
                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = subg(problem, options);  
                
                
            case {'SUBG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = subg(problem, options);                  
                
            case {'PG-BKT'}
                
                options.step_alg = 'backtracking';
                %options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);
                
            case {'PG-FIX'}
                
                options.step_alg = 'fix';
                options.step_alg = 'fix';
                options.step_init = 1/problem.L;                  
                %options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);                
                
                
            case {'PG-WOLFE'}
                
                options.step_alg = 'strong_wolfe';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);   
                
            case {'PG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);     
                
            case {'APG-BKT'}
                
                options.step_alg = 'backtracking';
                options.step_init_alg = 'bb_init';
                %[w_list{alg_idx}, info_list{alg_idx}] = sd_nesterov(problem, options);
                [w_list{alg_idx}, info_list{alg_idx}] = ag(problem, options);
                
            case {'APG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                %[w_list{alg_idx}, info_list{alg_idx}] = sd_nesterov(problem, options); 
                [w_list{alg_idx}, info_list{alg_idx}] = ag(problem, options);
                
            case {'FISTA'}
                
                options.sub_mode  = 'FISTA';                
                [w_list{alg_idx}, info_list{alg_idx}] = ista(problem, options); 
                
            case {'ADMM-LASSO'}
                
                options.rho = 0.1;
                [w_list{alg_idx}, info_list{alg_idx}] = admm_lasso(problem, options);    
                
            case {'CD-LASSO'}
                
                options.sub_mode = 'lasso';
                [w_list{alg_idx}, info_list{alg_idx}] = cd_lasso_elasticnet(problem, options);                   
                
            case {'L-BFGS-BKT'}
                
                options.step_alg = 'backtracking';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);
                
            case {'L-BFGS-WOLFE'}
                
                options.step_alg = 'strong_wolfe';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);  
                
            case {'P-Newton-CHOLESKY'}
                
                options.sub_mode = 'CHOLESKY';
                options.step_init_alg = 'bb_init';
                options.step_alg = 'tfocs_backtracking';
                %options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);  
                
            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
        
    end
    
    
    fprintf('\n\n');
    
    
    %% plot all
   
    % display iter vs cost
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display time vs cost
    display_graph('time','cost', algorithms, w_list, info_list);    
    % display iter vs. l1 norm, i.e. the toral number of non-zero elements 
    display_graph('iter','reg', algorithms, w_list, info_list); 
    
end




