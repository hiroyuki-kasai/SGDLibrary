function  test_linear_regression()

    clc;
    clear;
    close all;
    
    %rng('default')

    
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else           
        algorithms = gd_solver_list('NCG'); 
        %algorithms = {'NCG-FR-BTK','NCG-FR-WOLFE','NCG-PR-BTK','NCG-PR-WOLFE'};
    end

     
    %% prepare dataset
    if 1
        % generate synthtic data
        % sample data generating for training: y = w1*x1 + w2*x2 + ... * wd*1
        n = 10000;
        d = 100;
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
    problem = linear_regression(x_train, y_train, x_test, y_test, lambda);

    
    % calculate solution
    % solution
    w_opt = pinv(x_train * x_train') * x_train * y_train';
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt); 
    
    % initialize
    w_init = randn(d,1);

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
        options.f_opt = f_opt;        
        options.store_w = true;

        switch algorithms{alg_idx}
            case {'SD-STD'}
                
                options.step_alg = 'fix';
                options.step_init = 1;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

            case {'SD-BKT'}
                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

            case {'SD-EXACT'}
                
                options.step_alg = 'exact';                
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);
                
            case {'SD-WOLFE'}
                
                options.step_alg = 'strong_wolfe';
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);                
                
            case {'SD-SCALE-EXACT'}
                
                options.sub_mode = 'SCALING';
                options.step_alg = 'exact';                
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);
                
            case {'Newton-STD'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'Newton-DAMP'}

                options.sub_mode = 'DAMPED';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'Newton-CHOLESKY'}

                options.sub_mode = 'CHOLESKY';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);                

            case {'CG-PRELIM'}
                
                options.sub_mode = 'PRELIM';
                options.step_alg = 'exact';                   
                %options.beta_alg = 'PR';
                [w_list{alg_idx}, info_list{alg_idx}] = cg(problem, options);
                
            case {'CG-BKT'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'backtracking';      
                %options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = cg(problem, options);
                
            case {'CG-EXACT'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'exact';    
                %options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = cg(problem, options);
                
            case {'CG-PRECON-EXACT'}
                
                options.sub_mode = 'PRECON';
                % diagonal scaling
                options.M = diag(diag(A));                
                options.step_alg = 'exact';    
                options.beta_alg = 'PR';     
                
                [w_list{alg_idx}, info_list{alg_idx}] = cg(problem, options); 
                
            case {'NCG-FR-BTK'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'backtracking';      
                options.beta_alg = 'FR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);    
                
            case {'NCG-FR-WOLFE'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'strong_wolfe';      
                options.beta_alg = 'FR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);                  
                
            case {'NCG-PR-BTK'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'backtracking';      
                options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);    
                
            case {'NCG-PR-WOLFE'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'strong_wolfe';      
                options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);                   
             
            case {'BFGS-H-BKT'}
                
                options.step_alg = 'backtracking';                   
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);
                
            case {'BFGS-H-EXACT'}
                
                options.step_alg = 'exact';    
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);
                
            case {'BFGS-B-BKT'}
                
                options.step_alg = 'backtracking';     
                options.update_mode = 'B';
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);
                
            case {'BFGS-B-EXACT'}
                
                options.step_alg = 'exact';  
                options.update_mode = 'B';                
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);   
                
            case {'DAMPED-BFGS-BKT'}
                
                options.step_alg = 'backtracking';     
                options.update_mode = 'Damping';
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);
                
            case {'DAMPED-BFGS-EXACT'}
                
                options.step_alg = 'exact';  
                options.update_mode = 'Damping';                
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);    
                
            case {'L-BFGS-BKT'}
                
                options.step_alg = 'backtracking';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);
                
            case {'L-BFGS-EXACT'}
                
                options.step_alg = 'exact';    
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);  
                
            case {'L-BFGS-WOLFE'}
                
                options.step_alg = 'strong_wolfe';                  
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);                
                
            case {'BB'}
                
                options.step_alg = 'exact';    
                [w_list{alg_idx}, info_list{alg_idx}] = bb(problem, options);                
                
            case {'SGD'} 

                options.batch_size = 1;
                options.step = 0.1 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);   
                
            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
        
    end
    
    
    %% plot all
    close all;
    
    % display iter vs cost/gnorm
    display_graph('iter','cost', algorithms, w_list, info_list);
    display_graph('iter','gnorm', algorithms, w_list, info_list);  
    
    % draw convergence sequence
    w_history = cell(1);
    cost_history = cell(1);    
    for alg_idx=1:length(algorithms)    
        w_history{alg_idx} = info_list{alg_idx}.w;
        cost_history{alg_idx} = info_list{alg_idx}.cost;
    end    
    
    %return;
    draw_convergence_sequence(problem, w_opt, algorithms, w_history, cost_history);  
    
    % display regression results
    y_pred_list = cell(length(algorithms),1);
    mse_list = cell(length(algorithms),1);    
    for alg_idx=1:length(algorithms)    
        % predict class
        y_pred_list{alg_idx} = problem.prediction(w_list{alg_idx});
        % calculate accuracy
        mse_list{alg_idx} = problem.mse(y_pred_list{alg_idx}); 
    end 
    if plot_flag
        display_regression_result(problem, w_opt, algorithms, w_list, y_pred_list, mse_list, x_train, y_train, x_test, y_test);      
    end

end


