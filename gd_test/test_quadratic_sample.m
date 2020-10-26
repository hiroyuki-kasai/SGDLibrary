function  test_quadratic_sample()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    algorithms = {'SD-STD','SD-BKT','SD-EXACT','SD-SCALE-EXACT'};
     
    %% set A and b
    if 1
        % normal case
        % f = x(1)^2 + 2 * x(2)^2
        A = [1,0; 0,2];
        b = [0;0];
        w_init = [2; 1];
        
        func_sol = @(x) x(1)^2 + 2 * x(2)^2;
        
    elseif 1
        % ill-conditioned case
        % f = x(1)^2 + 0.001 * x(2)^2
        A = [1,0; 0,0.001];
        b = [0;0];
        w_init = [2; 1];
        
        func_sol = @(x) x(1)^2 + 0.001 * x(2)^2;        
    else
        % ill-conditioned case
        %f = 1000 * x(1)^2 + 40 * x(1) * x(2) + x(2)^2
        A = [1000,20; 20,1];
        b = [0;0];
        w_init = [1; 1000];    
        
        func_sol = @(x) 1000 * x(1)^2 + 40 * x(1) * x(2) + x(2)^2;       
    end
    fprintf('CN: %.4e\n', cond(A));
    
    
    %% define problem definitions
    problem = quadratic(A, b);
    
    
    % Calculate the solution
    fminunc_options.TolFun = 1e-36;    
    [w_opt,f_sol] = fminunc(func_sol, w_init, fminunc_options);
    fprintf('%f, %f, %f\n', w_opt(1), w_opt(2), f_sol);      

    % initialize
    w_list = cell(1);    
    info_list = cell(1);

    
    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_iter = 100;
        options.verbose = true;   
        options.f_sol = f_sol; 
        options.f_opt = f_sol;
        options.store_w = true;

        switch algorithms{alg_idx}
            
            case {'SD-NESTEROV'}
                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = ag(problem, options);
                
            case {'SD-STD'}
                
                options.step_alg = 'fix';
                options.step_init = 0.01;
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
                % diagonal scaling
                options.S = diag(1./diag(A));
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
                %options.beta_alg = 'PR';     
                
                [w_list{alg_idx}, info_list{alg_idx}] = cg(problem, options);    
             
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

end


