function [] = test_lasso()

    clc;
    clear;
    close all;
    
     
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = {'PG-BKT', 'PG-TFOCS-BKT', 'APG-BKT', 'APG-TFOCS-BKT', 'CD-LASSO', 'FISTA', 'ADMM-LASSO'}; 
%        algorithms = {'FISTA'};
    end    
    
    
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
        options.max_iter = 100;
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
                [w_list{alg_idx}, info_list{alg_idx}] = sd_nesterov(problem, options);
                
            case {'APG-TFOCS-BKT'}
                
                options.step_alg = 'tfocs_backtracking';
                options.step_init_alg = 'bb_init';
                [w_list{alg_idx}, info_list{alg_idx}] = sd_nesterov(problem, options); 
                
            case {'FISTA'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = fista(problem, options); 
                
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
    close all;
    
    % display iter vs cost/gnorm
    display_graph('iter','cost', algorithms, w_list, info_list);
    % display iter vs. l1 norm, i.e. the toral number of non-zero elements 
    display_graph('iter','reg', algorithms, w_list, info_list); 
    
end




