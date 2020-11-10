function  test_general()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    if 0
        algorithms = gd_solver_list('ALL');  
    else
        algorithms = gd_solver_list('BFGS'); 
        algorithms = {'L-BFGS-WOLFE','SD-STD','SD-BKT','Newton-STD','Newton-DAMP','Newton-CHOLESKY','NCG-BTK'};
        %algorithms = {'TR-NEWTON','TR-QUASI-NEWTON','L-BFGS-WOLFE'};
        algorithms = {'TR-DOGLEG-QUASI-NEWTON','TR-CAUCHY-QUASI-NEWTON','TR-TRUSTONE-QUASI-NEWTON','TR-CG-QUASI-NEWTON','L-BFGS-WOLFE'}; 
        algorithms = {'SD-BKT', 'NCG-BTK','TR-DOGLEG-QUASI-NEWTON','L-BFGS-BKT', 'BFGS-H-BKT','BB'};  
    end

     
    %% set cost, gradient and hessian
    if 0
        % f = x(1)^2 + 2 * x(2)^2
        % This is equivalent to the quadratic case wheree A = [1,0; 0,2] and b = [0;0].
        d = 2;
        f = @(x) x(1)^2 + 2 * x(2)^2;
        g = @(x) [2 * x(1); 4 * x(2)];   
        h = @(x) [2 0; 0 4];  
        w_init = [2; 1];
        
    elseif 0
        % f = 100 * x(1)^4 + 0.01 * x(2)^4
        d = 2;
        f = @(x) 100 * x(1)^4 + 0.01 * x(2)^4;
        g = @(x) [400 * x(1)^3; 0.04 * x(2)^3];   
        h = @(x) [1200 * x(1)^2, 0; 0 0.12 * x(2)^2];  
        w_init = [1; 1];     
        
    elseif 0
        % f = sqrt(1+(1).^2) + sqrt(1+x(2).^2)
        d = 2;
        f = @(x) sqrt(1+x(1).^2) + sqrt(1+x(2).^2);
        g = @(x) [x(1)/sqrt(x(1).^2+1); x(2)/sqrt(x(2).^2+1)];   
        h = @(x) [1/(x(1).^2+1).^1.5,0; 0, 1/(x(2).^2+1).^1.5];  
        w_init = [10; 10]; 
        
    elseif 1
        d = 2;
        f = @(x)7/5+(x(1)+2*x(2)+2*x(1)*x(2)-5*x(1)^2-5*x(2)^2)/(5* exp(x(1)^2+x(2)^2));
        g = @(x)[(1+2*x(2)-10*x(1)-2*x(1)*(x(1)+ 2*x(2)+2*x(1)*x(2)-5*x(1)^2 -5*x(2)^2))/(5* exp(x(1)^2+x(2)^2)); 
                (2+2*x(1)-10*x(2)-2*x(2)*(x(1)+2*x(2)+ 2*x(1)*x(2)-5*x(1)^2 -5*x(2)^2))/(5* exp(x(1)^2+x(2)^2))];
        h = @(x) [];
                
        w_init = [0;0.5];
        
    end
    
    
    %% define problem definitions
    problem = general(f, g, h, [], d, [], [], [], [], [], [], []);
    
    
    % Calculate the solution
    fminunc_options.TolFun = 1e-36;
    [w_opt,f_opt] = fminunc(f, w_init, fminunc_options);
    fprintf('%f, %f, %f\n', w_opt(1), w_opt(2), f_opt);      

    
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
        options.f_opt = f_opt;        
        options.store_w = true;

        switch algorithms{alg_idx}
            case {'TR-DOGLEG-NEWTON'}
                
                options.H_mode = 'NEWTON';
                [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options);  
                
            case {'TR-DOGLEG-QUASI-NEWTON'}
                
                options.H_mode = 'QUASI-NEWTON';
                options.subprob_solver = 'DOGLEG';
                [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options);   
                
            case {'TR-CAUCHY-QUASI-NEWTON'}
                
                options.H_mode = 'QUASI-NEWTON';
                options.subprob_solver = 'CAUCHY';
                [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options);  
                
            case {'TR-TRUSTONE-QUASI-NEWTON'}
                
                options.H_mode = 'QUASI-NEWTON';
                options.subprob_solver = 'TRUSTONE';
                [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options); 
                
            case {'TR-CG-QUASI-NEWTON'}
                
                options.H_mode = 'QUASI-NEWTON';
                options.subprob_solver = 'CG';
                [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options);       
                
%             case {'TR-Lanczos-QUASI-NEWTON'}
%                 
%                 options.H_mode = 'QUASI-NEWTON';
%                 options.subprob_solver = 'Lanczos';
%                 [w_list{alg_idx}, info_list{alg_idx}] = tr(problem, options);                 

                
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
    display_graph('iter','optimality_gap', algorithms, w_list, info_list);
    display_graph('time','optimality_gap', algorithms, w_list, info_list);
    %display_graph('iter','gnorm', algorithms, w_list, info_list);  
    
    % draw convergence sequence
    w_history = cell(1);
    cost_history = cell(1);    
    for alg_idx=1:length(algorithms)    
        w_history{alg_idx} = info_list{alg_idx}.w;
        cost_history{alg_idx} = info_list{alg_idx}.cost;
    end    
    draw_convergence_sequence(problem, w_opt, algorithms, w_history, cost_history);  

end


