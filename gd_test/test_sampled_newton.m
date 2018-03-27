function test_sampled_newton()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    algorithms = {'L-BFGS-BKT','Newton-CHOLESKY','Newton-INEXACT','Subsamp-Newton-Uniform', 'Subsamp-Newton-RNS', 'Subsamp-Newton-LS'};
    


    % select problem
    problem_type = 'log_reg';
    %problem_type = 'lin_reg';
    
    lambda = 0.01;
     
    % prepare datasets
    if strcmp(problem_type, 'lin_reg')
        
        % generate synthtic data
        % sample data generating for training: y = w1*x1 + w2*x2 + ... * wd*1
        n = 10000;
        d = 10;
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


        % define problem definitions
        problem = linear_regression(x_train, y_train, x_test, y_test, lambda);    
        f_opt1 = problem.cost(w_opt);
        fprintf('f_opt: %.16e\n', f_opt1);
        
        w_opt = problem.calc_solution(100, 'lbfgs');
        f_opt2 = problem.cost(w_opt); 
        fprintf('f_opt1: %.16e, f_opt2: %.16e, diff: %.16e\n', f_opt1, f_opt2, f_opt2 - f_opt1);    
        
        if f_opt2 < f_opt1
            f_opt = f_opt2;
        else
            f_opt = f_opt1;
        end
    
    elseif strcmp(problem_type, 'log_reg')
        
        % read real-world dataset
        [data_y, data_X] = libsvmread('../data/libsvm/a9a');
        x_train = data_X';
        y_train = data_y';             
        d = size(x_train,1);
        n = length(y_train);
        lambda = 0.5;
   
        if d == 0
            return;
        end
        
        % define problem definitions
        problem = logistic_regression(x_train, y_train, [], [], lambda);

        % calculate f_opt
        w_opt = problem.calc_solution(100, 'lbfgs');
        f_opt = problem.cost(w_opt); 
        fprintf('f_opt: %.24e\n', f_opt); 
        
    else
        return;
    end

    
    %% initialize
    w_init = zeros(d,1);
    batch_size = 1;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);       

    
    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_optgap = 1e-14;
        options.tol_gnorm = 1e-16;
        options.max_iter = 100;
        options.verbose = 1;   
        options.f_opt = f_opt;        
        options.store_w = false;
        options.permute_on = 1;    
        options.f_opt = f_opt; 
        options.batch_size = batch_size;
        

        switch algorithms{alg_idx}
            
            % Sub-sampled Newton methods
            case {'Subsamp-Newton-Uniform'}

                options.sub_mode = 'Uniform'; 
                options.subsamp_hess_size = 100*d;
                [w_list{alg_idx}, info_list{alg_idx}] = subsamp_newton(problem, options); 
                
            case {'Subsamp-Newton-RNS'}

                options.sub_mode = 'RNS'; 
                options.subsamp_hess_size = 20*d;
                [w_list{alg_idx}, info_list{alg_idx}] = subsamp_newton(problem, options);   
                
            case {'Subsamp-Newton-LS'}

                options.sub_mode = 'LS'; 
                options.subsamp_hess_size = 20*d;
                options.hess_update_freq = 10;
                [w_list{alg_idx}, info_list{alg_idx}] = subsamp_newton(problem, options);                       
            
            % Newton methodss
            case {'Newton-STD'}
                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'Newton-DAMP'}

                options.sub_mode = 'DAMPED';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);           
            
            case {'Newton-CHOLESKY-BKT'}

                options.sub_mode = 'CHOLESKY';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);
                
            case {'Newton-CHOLESKY'}

                options.sub_mode = 'CHOLESKY';                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);                
                
            case {'Newton-INEXACT-BKT'}

                options.sub_mode = 'INEXACT';   
                options.step_alg = 'backtracking';                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);       
                
            case {'Newton-INEXACT'}

                options.sub_mode = 'INEXACT';                
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);                      
                
            % BFGS variants
            case {'BFGS-BKT'}
                
                options.step_alg = 'backtracking';                  
                [w_list{alg_idx}, info_list{alg_idx}] = bfgs(problem, options);                    
                
            case {'L-BFGS-BKT'}
                
                options.step_alg = 'backtracking';  
                options.mem_size = 50;
                [w_list{alg_idx}, info_list{alg_idx}] = lbfgs(problem, options);  
                
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
    display_graph('iter','optimality_gap', algorithms, w_list, info_list); 
    display_graph('time','optimality_gap', algorithms, w_list, info_list);    
    %display_graph('time','gnorm', algorithms, w_list, info_list);  
     

end


