function [w, infos] = gd_nesterov(problem, options)
% Full gradient descent algorithm with Nesterov acceleration.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Nov. 01, 2016
% Modified by H.Kasai on Apr. 17, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  


    % extract options
    if ~isfield(options, 'step_init')
        step_init = 0.1;
    else
        step_init = options.step_init;
    end
    step = step_init;
    
    if ~isfield(options, 'step_alg')
        step_alg = 'backtracking';
    else
        step_alg  = options.step_alg;
    end  
   
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end      
    
    if ~isfield(options, 'tol_gnorm')
        tol_gnorm = 1.0e-12;
    else
        tol_gnorm = options.tol_gnorm;
    end    
    
    if ~isfield(options, 'max_iter')
        max_iter = 100;
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end   
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end    
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end    
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'STANDARD';
    else
        sub_mode = options.sub_mode;
    end  
    
    if ~isfield(options, 'S')
        S = eye(d);
    else
        S = options.S;
    end  
    
    ls_options.sub_mode = sub_mode;    
    if strcmp(sub_mode, 'STANDARD')
        %
    elseif strcmp(sub_mode, 'SCALING')
        ls_options.S = S;
    else
        %
    end    
    
    if ~isfield(options, 'use_restart')
        use_restart = true;
    else    
        use_restart = options.use_restart;
    end
    
    if ~isfield(options, 'step_init_alg')
        % Do nothing
    else
        if strcmp(options.step_init_alg, 'bb_init')
            % initialize by BB step-size
            step = bb_init(problem, w);
        end
    end     

    
    % initialise
    iter = 0;
    theta = 1;
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if isfield(problem, 'reg')
        infos.reg = problem.reg(w);   
    end
    if store_w
        infos.w = w;       
    end
    
    y = w;
    
    % set start time
    start_time = tic();  
    
    % print info
    if verbose
        if ~isfield(problem, 'prox')
            fprintf('AG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        else
            fprintf('APG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        end
    end     

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)      
        
        w_old = w;
        y_old = y;

        w = y - step*grad;   
        
        % proximal operator
        if isfield(problem, 'prox')
            w = problem.prox(w, step);
        end        
        
        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap];        
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if isfield(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            if ~isfield(problem, 'prox')
                fprintf('AG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
            else
                fprintf('APG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
            end                
        end    
        
        % calculate nesterov step
        theta = 2/(1 + sqrt(1+4/(theta^2)));
        
        if (use_restart && (y-w)'*(w-w_old)>0)
            w = w_old;
            y = w;
            theta = 1;
        else
            y = w + (1-theta)*(w-w_old);
        end
        
        % calculate gradient
        grad_old = grad;        
        grad = problem.full_grad(y);
  

        % line search
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, -grad, w, rho, c);
        elseif strcmp(step_alg, 'exact')
            step = exact_line_search(problem, 'GD', -grad, [], [], w, ls_options);
        elseif strcmp(step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, -grad, w, c1, c2);
        elseif strcmp(step_alg, 'tfocs_backtracking') 
            c1 = 1.01;
            c2 = 0.5; 
            step = tfocs_backtracking_search(step, y, y_old, grad, grad_old, c1, c2);
        else
        end
        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end
