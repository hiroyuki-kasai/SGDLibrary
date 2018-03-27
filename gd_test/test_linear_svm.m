function  test_linear_svm()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else     
        %algorithms = gd_solver_list('LS');
        %algorithms = gd_solver_list('NCG');        
        %algorithms = gd_solver_list('BFGS'); 
        algorithms = {'SD-STD', 'SD-BKT'}; 
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
                
            case {'NCG-BTK'}
                
                options.sub_mode = 'STANDARD';                
                options.step_alg = 'backtracking';      
                options.beta_alg = 'PR';                
                [w_list{alg_idx}, info_list{alg_idx}] = ncg(problem, options);    
                
            case {'NCG-WOLFE'}
                
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
    draw_convergence_sequence(problem, w_opt, algorithms, w_history, cost_history);     
    
    % display classification results
    y_pred_list = cell(length(algorithms),1);
    accuracy_list = cell(length(algorithms),1);    
    for alg_idx=1:length(algorithms)  
        p = problem.prediction(w_list{alg_idx});
        % calculate accuracy
        accuracy_list{alg_idx} = problem.accuracy(p); 
        
        fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, problem.accuracy(p));        
        
        % convert from {1,-1} to {1,2}
        p(p==-1) = 2;
        p(p==1) = 1;
        % predict class
        y_pred_list{alg_idx} = p;
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


