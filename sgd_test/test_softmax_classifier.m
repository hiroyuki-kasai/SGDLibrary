function  test_softmax_classifier()
        
    clc;
    clear;
    close all;

    
    %% Set algorithms
    % Note that 'Reg-oBFGS-Inf', 'oBFGS-Inf' and 'Damp-oBFGS-Inf' do not work due to memory limitation.
    % Note that 'SQN','SVRG-SQN','SVRG-LBFGS' and 'SS-SVRG' are not suppoted. 
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SGD','SVRG','Adam','IQN'};     
    end      
    
    
    if ismember('Reg-oBFGS-Inf', algorithms) || ismember('oBFGS-Inf', algorithms) || ismember('Damp-oBFGS-Inf', algorithms)
        fprintf('Reg-oBFGS-Inf, oBFGS-Inf and Damp-oBFGS-Inf do not work properly due to memory limitation. \nPlease reconfigure proper algorithms, and execute this script again.\nThank you.\n');
        return;
    end    
    
    if ismember('SQN', algorithms) || ismember('SVRG-SQN', algorithms) || ismember('SVRG-LBFGS', algorithms)
        fprintf('SQN, SVRG-SQN and SVRG-LBFGS are not supported in this problem. \nPlease reconfigure proper algorithms, and execute this script again.\nThank you.\n');
        return;
    end
    
    
    %% prepare dataset
    if 1
        n_per_class = 100;    % # of samples        
        d = 3;      % # of dimensions     
        l = 5;      % # of classes 
        std = 0.15; % standard deviation

        data = multiclass_data_generator(n_per_class, d, l, std);  
        n = length(data.y_train);
        d = d + 1; % adding '1' row for intersect
        
        % train data        
        x_train = [data.x_train; ones(1,n)];
        % transform class label into label logical matrix
        y_train = zeros(l,n);
        for j=1:n
            y_train(data.y_train(j),j) = 1;
        end        

        % test data
        x_test = [data.x_test; ones(1,n)];
        % transform class label into label logical matrix
        y_test = zeros(l,n);
        for j=1:n
            y_test(data.y_test(j),j) = 1;
        end     
        
        lambda = 0.0001;
        w_opt = zeros(d*l,1);            
        
    else
        % load real-world data
        data = importdata('../data/mnist/6000_data_0.001.mat');
        x_train = data.x_trn;
        y_train = data.y_trn; 
        x_test = data.x_tst;
        y_test= data.y_tst;         
        d = size(x_train,1);
        n = length(y_train);
        lambda = data.lambda;
        
        w_opt = data.w_opt;
        l = data.L;
    end
    
    % set plot_flag
    if d > 4
        plot_flag = false;  % too high dimension  
    else
        plot_flag = true;
    end       

    
    %% define problem definitions
    problem = softmax_regression(x_train, y_train, x_test, y_test, l, lambda);

   
    %% initialize
    w_init = randn(d*l,1);
    batch_size = 10;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);
    algname_list = cell(length(algorithms),1);    
    
    
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
        options.verbose = true;
        options.lambda = lambda;
        options.permute_on = 1; 
        options.f_opt = f_opt;
        
        switch algorithms{alg_idx}
            case {'SD'}
                
                options.step_init = 2;
                options.max_iter = 30 * options.max_epoch;
                [w_list{alg_idx}, info_list{alg_idx}] = sd(problem, options);

            case {'SGD'} 

                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);   
                
            % Variance reduction (VR) varitns                   
            case {'SVRG'}
                
                options.batch_size = batch_size;
                options.step_init = 0.5 * options.batch_size;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);      
                
            case {'SAG'}
                
                options.batch_size = batch_size;
                %options.step_init = 0.00005 * options.batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);   
                
            case {'SAGA'}
                
                options.batch_size = batch_size;
                %options.step_init = 0.00005 * options.batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAGA';                       

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);                  
                
            % AdaGrad variants                
            case {'AdaGrad'}
                
                options.batch_size = batch_size;
                options.step_init = 0.01 * options.batch_size;
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
                options.step_init = 0.01 * options.batch_size;
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
                options.step_init = 0.00001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SQN';        
                options.l = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);

            case {'SLBFGS-SVRG'}                  
 
                options.batch_size = batch_size;
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG';
                options.l = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);

            case {'oBFGS-Inf'} 

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = false;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);

            case {'oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.r = 20;
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
                options.step_init = 0.00000001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Lim-mem';
                options.r = 20;
                options.regularized = true;  
                options.delta = 0.1;     

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);
                
            case {'Damp-oBFGS-Inf'}

                options.batch_size = batch_size;
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-mem';
                options.regularized = true;  
                options.delta = 0.1;
                options.damping = true;

                [w_list{alg_idx}, info_list{alg_idx}] = obfgs(problem, options);         
                
            case {'Damp-oBFGS-Lim'}

                options.batch_size = batch_size;
                options.step_init = 0.001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'Inf-Lim';
                options.regularized = true;  
                options.delta = 0.1;
                options.damping = true;

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
        
        algname_list{alg_idx} = algorithms{alg_idx};
    end
    
    fprintf('\n\n');

    
    
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
            % predict class
            y_pred_list{alg_idx} = problem.prediction(w_list{alg_idx});
            % calculate accuracy
            accuracy_list{alg_idx} = problem.accuracy(y_pred_list{alg_idx}); 
            fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, accuracy_list{alg_idx});       
        else
            fprintf('Classificaiton accuracy: %s: Not supported\n', algorithms{alg_idx}); 
        end
    end      

    % convert logial matrix to class label vector
    [~, y_train] = max(y_train, [], 1);
    [~, y_test] = max(y_test, [], 1);    
    if plot_flag
        display_classification_result(problem, algorithms, w_list, y_pred_list, accuracy_list, x_train, y_train, x_test, y_test);  
    end
    
end


