function  test_l1_linear_regression()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SVRG', 'SAG', 'Adam', 'APG-BKT', 'APG-TFOCS-BKT'};
    end    

     
    %% prepare dataset
    if 1
        % generate synthtic data
        % sample data generating for training: y = w1*x1 + w2*x2 + ... * wd*1
        n = 100;
        d = 2;
        std = 0.25;
        data = linear_regression_data_generator(n, d, std);
        
        x_train = data.x_train;
        y_train = data.y_train;    
        x_test = data.x_test;
        y_test = data.y_test;           
        
    elseif 0
        % load real-world data        
        data = load('../data/linear_regression/ex1data2.txt');
        x_in = data(:,1:2);
        y_in = data(:,3);
        n = length(y_in);
        n_train = floor(n/2);
        d = 2;
        
        % scale features and set them to zero mean
        x_in = featureNormalize(x_in);
        % add intercept term to x_in
        x_in = [x_in ones(n, 1)];      
        
        x_train = x_in(1:n_train,:)';
        y_train = y_in(1:n_train)';     
        x_test = x_in(n_train+1:end,:)';
        y_test = y_in(n_train+1:end)';           
        
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
        
    end
    
    % for intersect
    d = d + 1;          
    lambda = 0.01;   
    
    % set plot_flag
    if d > 4
        plot_flag = false;  % too high dimension  
    else
        plot_flag = true;
    end      

    
    %% define problem definitions
    problem = l1_linear_regression(x_train, y_train, x_test, y_test, lambda);

    
    %% initialize
    w_init = randn(d,1);
    batch_size = 10;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    
    
    %% calculate solution
    options.w_init = w_init;
    options.max_iter = 1000;
    w_opt = problem.calc_solution('ag', options);    
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);       
    
    
    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol = 10^-24;
        options.max_epoch = 500;
        options.verbose = true;
        options.lambda = lambda;
        options.permute_on = 1; 
        options.f_opt = problem.cost(w_opt);        
        

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
                
            case {'SD'}
                
                options.step_init = 0.1;
                options.max_iter = 10 * options.max_epoch;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

                w_opt = w_list{alg_idx};

            case {'SGD'} 

                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);   
                
            % Variance reduction (VR) varitns                   
            case {'SVRG'}
                
                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
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
    
    
    %% plot all
    close all;
    % display cost vs grads
    display_graph('grad_calc_count','cost', algorithms, w_list, info_list);
    % display optimality gap vs grads
    if options.f_opt ~= -Inf
        display_graph('grad_calc_count','optimality_gap', algorithms, w_list, info_list);
    end
    
    % display regression results
    y_pred_list = cell(length(algorithms),1);
    mse_list = cell(length(algorithms),1);    
    for alg_idx=1:length(algorithms)    
        if ~isempty(w_list{alg_idx})
            % predict class
            y_pred_list{alg_idx} = problem.prediction(w_list{alg_idx});
            % calculate accuracy
            mse_list{alg_idx} = problem.mse(y_pred_list{alg_idx}); 
        end
    end 
    if plot_flag
        display_regression_result(problem, w_opt, algorithms, w_list, y_pred_list, mse_list, x_train, y_train, x_test, y_test);      
    end

end


