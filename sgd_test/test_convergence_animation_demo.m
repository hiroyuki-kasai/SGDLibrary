function  test_convergence_animation_demo()

    clc;
    clear;
    close all;


    %% Set algorithms
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SGD','SVRG','SQN','IQN','Damp-oBFGS-Lim','AdaGrad'};
    end      
    

     
    %% prepare dataset
    if 1
        n = 1000;
        d = 1;
        std = 0.55;
        
        % generate data
        data = linear_regression_data_generator(n, d, std);
        % set train data        
        x_train = data.x_train;
        y_train = data.y_train;  
        % set test data        
        x_test = data.x_test;
        y_test = data.y_test;     
        % set solution
        w_opt = pinv(x_train * x_train') * x_train * y_train'
        % for intersect  
        d = d + 1;
        % set lambda         
        lambda = 0.01;   

        % define problem definitions
        problem = linear_regression(x_train, y_train, x_test, y_test, lambda);            
        
    elseif 0
        n = 300;        
        d = 2;

        % generate data
        data = logistic_regression_data_generator(n, d);
        % set train data
        x_train = data.x_train;
        y_train = data.y_train;  
        % set test data
        x_test = data.x_test;
        y_test = data.y_test; 
        % set solution        
        w_opt = data.w_opt;    
        % set lambda 
        lambda = 0.1;
    
        % define problem definitions
        problem = logistic_regression(x_train, y_train, x_test, y_test, lambda);   
        
    else
        l = 2;
        n = 100;    % # of samples per class           
        d = 1;      % # of dimensions
        std = 0.15; % standard deviation        
        
        % generate data        
        data = multiclass_data_generator(n, d, l, std);
        d = d + 1; % adding '1' row for intersect
        
        % train data        
        x_train = [data.x_train; ones(1,l*n)];
        % assign y (label) {1,-1}
        y_train(data.y_train<=1.5) = -1;
        y_train(data.y_train>1.5) = 1;
        % test data
        x_test = [data.x_test; ones(1,l*n)];
        % assign y (label) {1,-1}        
        y_test(data.y_test<=1.5) = -1;
        y_test(data.y_test>1.5) = 1;
        % set solution              
        w_opt = zeros(d,1);          
        % set lambda         
        lambda = 0.1;
 
        % define problem definitions
        problem = linear_svm(x_train, y_train, x_test, y_test, lambda);    
        
    end
    

    %% initialize
    w_init = randn(d,1);
    batch_size = 10;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);

    
    %% calculate solution
    if norm(w_opt)
    else
        % calculate solution
        w_opt = problem.calc_solution(1000);
    end
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);       
    
    
    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol = 10^-24;
        options.max_epoch = 100;
        options.verbose = 2;
        options.lambda = lambda;
        options.permute_on = 1; 
        options.f_opt = problem.cost(w_opt);
        options.store_w = true;
        

        switch algorithms{alg_idx}
            case {'SD'}
                
                options.step_init = 0.1;
                options.max_iter = 10 * options.max_epoch;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

                w_opt = w_list{alg_idx};

            case {'SGD'} 

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
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
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAG';                   

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);
                
            case {'SAGA'}
                
                options.batch_size = batch_size;
                %options.step_init = 0.00005 * options.batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAGA';                       

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);                   
                
            % AdaGrad variants                
            case {'AdaGrad'}
                
                options.batch_size = batch_size;
                options.step_init = 0.05 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'AdaGrad';        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
    
            case {'RMSProp'}    
    
                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'RMSProp';
                options.beta = 0.9;

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);

            case {'AdaDelta'}                  
    
                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;

                options.sub_mode = 'AdaDelta';     
                options.beta = 0.9;        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
   
            case {'Adam'}                 

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Adam';
                options.beta1 = 0.8;
                options.beta2 = 0.999;
                options.epsilon = 0.00001;

                [w_list{alg_idx}, info_list{alg_idx}] = adam(problem, options);
                
            case {'AdaMax'}                 

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
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
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-SQN';
                options.L = 20;
                options.mem_size = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);
                
            case {'SVRG-LBFGS'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.001 * options.batch_size;
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
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = false;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.mem_size = 20;
                options.regularized = false;        

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Inf'}

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.mem_size = 20;
                options.regularized = true;  
                options.delta = 0.1;     

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);
                
            case {'Damp-oBFGS-Inf'} % SDBFGS

                options.batch_size = batch_size;
                options.step_init = 0.005 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;
                options.damped = true;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);  
                
                
            case {'Damp-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.005 * options.batch_size;
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
    %display_graph('grad_calc_count','cost', algorithms, w_list, info_list);
    % display optimality gap vs grads
    if options.f_opt ~= -Inf
        %display_graph('grad_calc_count','optimality_gap', algorithms, w_list, info_list);
    end
    
    % draw convergence animation
    w_history = cell(1,1);
    for alg_idx=1:length(algorithms)    
        if ~isempty(w_list{alg_idx}) 
            w_history{alg_idx} = info_list{alg_idx}.w;
        else
            w_history{alg_idx} = ones(d,1);
        end
    end
    speed = 0.5;
    draw_convergence_animation(problem, algorithms, w_history, options.max_epoch, speed);    

end


