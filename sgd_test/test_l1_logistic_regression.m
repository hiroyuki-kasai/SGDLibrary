function [] = test_l1_logistic_regression()

    clc;
    clear;
    close all;
    
    %rng('default');
    
     
    %% Set algorithms
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SVRG', 'SAG', 'Adam', 'Reg-oBFGS-Lim','APG-BKT', 'APG-TFOCS-BKT'};
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
        %w_opt = data.w_opt;        
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
    
    
    %% initialize
    w_init = rand(d,1); 
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
        

    %% calculate solution
    %if norm(w_opt)
    %else
        % calculate solution
        options.w_init = w_init;
        options.max_iter = 100;
        w_opt = problem.calc_solution(options, 'ag');
    %end
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);  
    
    %% calculate solution
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);      
    
    


    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 100;
        options.max_epoch = 100;
        options.verbose = true;  
        options.f_opt = f_opt;
        batch_size = 10;        

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
                
                
            case {'SGD'} 

                options.batch_size = batch_size;
                options.step = 0.001 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);                   
                
            % Variance reduction (VR) varitns                   
            case {'SVRG'}
                
                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);      
                
            case {'SAG'}
                
                options.batch_size = batch_size;
                %options.step_init = 0.00005 * options.batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAG';                   

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);
                
            case {'SAGA'}
                
                options.batch_size = batch_size;
                %options.step_init = 0.00005 * options.batch_size;
                options.step_init = 0.000001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAGA';                       

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);                   
                
            % AdaGrad variants                
            case {'AdaGrad'}
                
                options.batch_size = batch_size;
                options.step_init = 0.02 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'AdaGrad';        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
    
            case {'RMSProp'}    
    
                options.batch_size = batch_size;
                options.step_init = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'RMSProp';
                options.beta = 0.9;

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);

            case {'AdaDelta'}                  
    
                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;

                options.sub_mode = 'AdaDelta';     
                options.beta = 0.9;        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
   
            case {'Adam'}                 

                options.batch_size = batch_size;
                options.step_init = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Adam';
                options.beta1 = 0.8;
                options.beta2 = 0.999;
                options.epsilon = 0.00001;

                [w_list{alg_idx}, info_list{alg_idx}] = adam(problem, options);
                
            case {'AdaMax'}                 

                options.batch_size = batch_size;
                options.step_init = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'AdaMax';
                options.beta1 = 0.8;
                options.beta2 = 0.999;
                options.epsilon = 0.00001;

                [w_list{alg_idx}, info_list{alg_idx}] = adam(problem, options);                
                
            
            % Stochastic Quasi-Newton variants
            case {'SQN'}             

                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SQN';        
                options.L = 20;
                options.mem_size = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);

            case {'SVRG-SQN'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-SQN';
                options.L = 20;
                options.mem_size = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);
                
            case {'SVRG-LBFGS'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-LBFGS';
                options.mem_size = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);  
                
            case {'SS-SVRG'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.0005 * options.batch_size;
                options.step_alg = 'fix';
                r = d-1; 
                if r < 1
                    r = 1;
                end
                options.r = r;

                [w_list{alg_idx}, info_list{alg_idx}] = subsamp_svrg(problem, options);                    

            case {'oBFGS-Inf'} 

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = false;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.mem_size = 20;
                options.regularized = false;        

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Inf'}

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.mem_size = 20;
                options.regularized = true;  
                options.delta = 0.1;     

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);
                
            case {'Damp-oBFGS-Inf'} % SDBFGS

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;
                options.damped = true;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);  
                
                
            case {'Damp-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.regularized = true;  
                options.delta = 0.1;
                options.damped = true;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);     
                
                
           case {'IQN'} 

                options.w_init = w_init;
                options.step_init = 1;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = iqn(problem, options);                      
                
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
    

    display_graph('grad_calc_count','optimality_gap', algorithms, w_list, info_list);
   
    
end




