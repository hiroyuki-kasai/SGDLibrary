function [w, infos] = ncg(problem, options)
% Nonlinear conjugate gradient algorithm.
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
% Created by H.Kasai on Oct. 30, 2016


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
        store_w = true;
    else
        store_w = options.store_w;
    end    
    
    if ~isfield(options, 'beta_alg')
        beta_alg = 'FR';
    else
        beta_alg = options.beta_alg;
    end  
    
    if ~isfield(options, 'S')
        S = eye(d);
    else
        S = options.S;
    end
  
    if ~isfield(options, 'step_init_alg')
        % Do nothing
    else
        if strcmp(options.step_init_alg, 'bb_init')
            % initialize by BB step-size
            step_init = bb_init(problem, w);
        end
    end 
    

    % initialise
    iter = 0;
    stopping = false;
    w_corrupted = false;  
    
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
    
    % set start time
    start_time = tic(); 
    
    % print info
    if verbose
        fprintf('NCG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
    end      
    
    % set first direction
    grad_old = grad;         
    d_old = -grad;

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter) && ~stopping      
        
        % Revert to steepest descent if is not direction of descent                
        if (d_old'*grad_old > 0)
            d_old = -d_old;
        end        

        % line search
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, d_old, w, rho, c);
        elseif strcmp(step_alg, 'exact')
            ls_options.sub_mode = sub_mode;
            ls_options.S = S;
            step = exact_line_search(problem, 'GD', -grad, [], [], w, ls_options);            
        elseif strcmp(step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, d_old, w, c1, c2);
        elseif strcmp(step_alg, 'tfocs_backtracking') 
            if iter > 0
                alpha = 1.05;
                beta = 0.5; 
                step = tfocs_backtracking_search(step, w, w_old, grad, grad_old, alpha, beta);
            else
                step = step_init;
            end
        else            
        end
        
        % update w
        w_old = w;        
        w = w + step * S * d_old;
        
        % proximal operator
        if isfield(problem, 'prox')
            w = problem.prox(w, step);
        end          
        
        % store old info   
        grad_old = grad;   
        % calculate gradient
        grad = problem.full_grad(w);        
        
        % updata beta
        switch beta_alg
            case 'FR',
                beta = (grad'*S*grad)/(grad_old'*S*grad_old);
            case 'PR',
                beta = ((grad-grad_old)'*S*grad)/(grad_old'*S*grad_old);
            otherwise,
        end
        
        % avoid negative conjugate direction weights
        if beta < 0
            beta = max(0,beta);
        end        

        % update direction             
        d = -S * grad + beta * d_old;    
        
        % store d
        d_old = d;

        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        
        
        % store info
        if ~(any(isinf(w(:))) || any(isnan(w(:)))) && ~isnan(f_val) && ~isinf(f_val)     
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
        else
            w_corrupted = true;
            w = infos.w(:,end);
            stopping = true;            
        end        
       
        % print info
        if verbose
            fprintf('NCG: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    elseif w_corrupted
        fprintf('Solution corrupted\n');        
    end    
    
end
