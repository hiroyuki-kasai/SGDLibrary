function [w, infos] = cg(problem, options)
% Linear conjugate gradient (CG) algorithm.
%
% The algorithm of interest is defined as
%
%           min f(x) = 1/2 * x^T * A * x - b^T * x.
%           where 
%           x in R^d 
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% Reference:
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       Algorithm 5.1, 5.2 in Section 5.1.
%
%       sub_mode    'PRELIM'    Algorithm 5.1
%       sub_mode    'STANDARD'  Algorithm 5.2
%       sub_mode    'PRECON'    Algorithm 5.3
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 30, 2016
% Modified by H.Kasai on Oct. 31, 2016


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
    
    if ~isfield(options, 'step_init_alg')
        % Do nothing
    else
        if strcmp(options.step_init_alg, 'bb_init')
            % initialize by BB step-size
            step_init = bb_init(problem, w);
        end
    end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'STANDARD';
    else
        sub_mode = options.sub_mode;
    end      
    
    if ~isfield(options, 'M')
        if strcmp(sub_mode, 'PRECON')
            h = problem.full_hess(w);
            M = diag(diag(h));
        else
            M = eye(d);
        end
    else
        M = options.M;
    end  
    
    ls_options.sub_mode = sub_mode;    
    if strcmp(sub_mode, 'STANDARD')
        %
    elseif strcmp(sub_mode, 'PRECON')
        ls_options.M = M;
    else
        %
    end     
    

    % initialise
    iter = 0;
    
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
    
    
    % set first residual, i.e., r = Ax - b;
    r_old = grad;
    y_old = [];  
    
    % initialise
    if strcmp(sub_mode, 'PRELIM')
        % set directoin
        p = -r_old;        
    elseif strcmp(sub_mode, 'STANDARD')
        % set directoin
        p = -r_old;           
    elseif strcmp(sub_mode, 'PRECON')
        % solve y
        y_old = M \r_old;       
        % set directoin
        p = -y_old;           
    else
        fprintf('sub_mode %s is not supported\n', sub_mode);
        return;
    end    
    
    % set start time
    start_time = tic();  
    
    % print info
    if verbose
        fprintf('CG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
    end       
    
    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        
        
        % Revert to steepest descent if is not direction of descent                
        if (p'*grad > 0)
            p = -p;
        end        

        % line search
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, p, w, rho, c);
        elseif strcmp(step_alg, 'exact')
            ls_options.M = M;
            step = exact_line_search(problem, 'CG', p, r_old, y_old, w, ls_options);
        elseif strcmp(step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, -grad, w, c1, c2);
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
        w = w + step * p;
        
        % proximal operator
        if isfield(problem, 'prox')
            w = problem.prox(w, step);
        end                  
        
        % calculate gradient
        grad_old = grad;
        grad = problem.full_grad(w);   
        
        
        if strcmp(sub_mode, 'PRELIM')
            % updata residual 
            r = grad;
            
            % updata beta        
            beta = r' * problem.A() * p / (p' * problem.A() * p);
            
            % update direction  
            p = - grad + beta * p;         
        
        elseif  strcmp(sub_mode, 'STANDARD')
            % updata residual             
            r = r_old + step * problem.A() * p;
                        
            % updata beta        
            beta = r' * r / (r_old' * r_old);            
            
            % updata beta       
            p = - r + beta * p;  
            
            % store r
            r_old = r;            
        else
            % updata residual             
            r = r_old + step * problem.A() * p;
            
            % solve y
            y = M \r;
                        
            % updata beta        
            beta = r' * y / (r_old' * y_old);            
            
            % updata beta       
            p = -y + beta * p;  
            
            % store r
            r_old = r;  
            y_old = y;
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
            fprintf('CG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
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
