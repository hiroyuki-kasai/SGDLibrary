function  test_linear_svm()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SGD','SVRG','IQN'};
    end   
    

    % # of classes (must not change)
    l = 2;
     
    %% prepare dataset
    if 1     % generate synthetic data
        n = 100;    % # of samples per class           
        d = 3;      % # of dimensions
        std = 0.15; % standard deviation        
        
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
       
    else    % load real-world data
        data = importdata('../data/mushroom/mushroom.mat');
        n = size(data.X,1);
        d = size(data.X,2) + 1;         
        x_in = [data.X ones(n,1)]';
        y_in = data.y';
        
        perm_idx = randperm(n);
        x = x_in(:,perm_idx);
        y = y_in(perm_idx);        
        
        % split data into train and test data
        % train data
        n_train = floor(n/8);
        x_train = x(:,1:n_train);
        y_train = y(1:n_train);  
        x_train_class1 = x_train(:,y_train>0);
        x_train_class2 = x_train(:,y_train<0);  
        n_class1 = size(x_train_class1,2);
        n_class2 = size(x_train_class2,2);        
        
        % test data
        x_test = x(:,n_train+1:end);
        y_test = y(n_train+1:end);  
        x_test_class1 = x_test(:,y_test>0);
        x_test_class2 = x_test(:,y_test<0);  
        n_test_class1 = size(x_test_class1,2);
        n_test_class2 = size(x_test_class2,2);    
        n_test = n_test_class1 + n_test_class2;

    end
    lambda = 0.1;
    w_opt = zeros(d,1); 
    
    % set plot_flag
    if d > 4
        plot_flag = false;  % too high dimension  
    else
        plot_flag = true;
    end     

    
    %% define problem definitions
    problem = linear_svm(x_train, y_train, x_test, y_test, lambda);

    
    %% initialize
    w_init = randn(d,1);
    batch_size = 5;
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
        %options.tol = -Inf;
        options.max_epoch = 500;
        options.verbose = true;
        options.lambda = lambda;
        options.permute_on = 1; 
        options.f_opt = f_opt;        
        
        
        switch algorithms{alg_idx}
            case {'SD'}
                
                options.step = 0.05;
                options.max_iter = 10*options.max_epoch;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

            case {'SGD'} 

                options.batch_size = batch_size;
                options.step = 0.001 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);   
                
            % Variance reduction (VR) varitns                   
            case {'SVRG'}
                
                options.batch_size = batch_size;
                options.step = 0.01 * options.batch_size;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);      
                
            case {'SAG'}
                
                options.batch_size = batch_size;
                %options.step = 0.00005 * options.batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAG';                 

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);    
                
            case {'SAGA'}
                
                options.batch_size = batch_size;
                %options.step = 0.00005 * options.batch_size;
                options.step = 0.000001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAGA';                       

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);                    
                
            % AdaGrad variants                
            case {'AdaGrad'}
                
                options.batch_size = batch_size;
                options.step = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'AdaGrad';        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
    
            case {'RMSProp'}    
    
                options.batch_size = batch_size;
                options.step = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;
                options.sub_mode = 'RMSProp';
                options.beta = 0.9;

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);

            case {'AdaDelta'}                  
    
                options.batch_size = batch_size;
                options.step = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.epsilon = 0.00001;

                options.sub_mode = 'AdaDelta';     
                options.beta = 0.9;        

                [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
   
            case {'Adam'}                 

                options.batch_size = batch_size;
                options.step = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Adam';
                options.beta1 = 0.8;
                options.beta2 = 0.999;
                options.epsilon = 0.00001;

                [w_list{alg_idx}, info_list{alg_idx}] = adam(problem, options);
                
            case {'AdaMax'}                 

                options.batch_size = batch_size;
                options.step = 0.00001 * options.batch_size;
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
                options.step = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SQN';        
                options.L = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);

            case {'SVRG-SQN'}       
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-SQN';
                options.L = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);
                
            case {'SVRG-LBFGS'}       
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step = 0.01 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-LBFGS';
                options.L = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options); 
                
            case {'SS-SVRG'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step = 0.0005 * options.batch_size;
                options.step_alg = 'fix';
                r = d-1; 
                if r < 1
                    r = 1;
                end
                options.r = r;

                [w_list{alg_idx}, info_list{alg_idx}] = subsamp_svrg(problem, options);                    

            case {'oBFGS-Inf'} 

                options.batch_size = batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = false;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.r = 20;
                options.regularized = false;        

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Inf'}

                options.batch_size = batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'Reg-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.r = 20;
                options.regularized = true;  
                options.delta = 0.1;     

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);
                
            case {'Damp-oBFGS-Inf'}

                options.batch_size = batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;
                options.damped = true;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);   
                
            case {'Damp-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-Lim';
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
    
    % display classification results
    y_pred_list = cell(length(algorithms),1);
    accuracy_list = cell(length(algorithms),1);    
    for alg_idx=1:length(algorithms)  
        if ~isempty(w_list{alg_idx})        
            p = problem.prediction(w_list{alg_idx});
            % calculate accuracy
            accuracy_list{alg_idx} = problem.accuracy(p); 

            fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, problem.accuracy(p));        

            % convert from {1,-1} to {1,2}
            p(p==-1) = 2;
            p(p==1) = 1;
            % predict class
            y_pred_list{alg_idx} = p;
        else
            fprintf('Classificaiton accuracy: %s: Not supported\n', algorithms{alg_idx});   
        end
    end 
    
    % convert from {1,-1} to {1,2}
    y_train(y_train==-1) = 2;
    y_train(y_train==1) = 1;
    y_test(y_test==-1) = 2;
    y_test(y_test==1) = 1;    
    if plot_flag
        display_classification_result(problem, algorithms, w_list, y_pred_list, accuracy_list, x_train, y_train, x_test, y_test);
    end

end


