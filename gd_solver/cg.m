function [w, infos] = cg(problem, in_options)
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
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 30, 2016
% Modified by H.Kasai on Mar. 25, 2018
% Modified by H.Kasai on Oct. 22, 2020


    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = []; 
    local_options.algorithm = 'CG';    
    local_options.sub_mode = 'STANDARD';

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);  


    if ~isfield(options, 'M')
        if strcmp(options.sub_mode, 'PRECON')
            h = problem.full_hess(w);
            M = diag(diag(h));
        else
            M = eye(d);
        end
    else
        M = options.M;
    end  
    
%     ls_options.sub_mode = sub_mode;    
%     if strcmp(sub_mode, 'STANDARD')
%         %
%     elseif strcmp(optionssub_mode, 'PRECON')
%         ls_options.M = M;
%     else
%         %
%     end     
    

    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;    
    
   % initialize by BB step-size 
    if strcmp(options.step_init_alg, 'bb_init')
        options.step_init = bb_init(problem, w);
    end    
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    grad_old = [];
    
    % display infos
    if options.verbose
        fprintf('CG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, iter, f_val, gnorm, optgap);
    end  
    
    
    % set first residual, i.e., r = Ax - b;
    r_old = grad;
    y_old = []; 
    w_old = w;
    grad_old = grad;
    prev_step = options.step_init;    
    
    % initialise
    if strcmp(options.sub_mode, 'PRELIM')
        % set directoin
        p = -r_old;        
    elseif strcmp(options.sub_mode, 'STANDARD')
        % set directoin
        p = -r_old;           
    elseif strcmp(options.sub_mode, 'PRECON')
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
    
    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)       
        
        % Revert to steepest descent if is not direction of descent                
        if (p'*grad > 0)
            p = -p;
        end        

        % line search
        options.r_old = r_old;
        options.y_old = y_old;        
        [step, ~] = options.linesearchfun(options.step_alg, problem, w, w_old, p, grad_old, prev_step, options);   
        prev_step = step;
        
        % update w
        w_old = w;
        w = w + step * p;
        
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end                  
        
        % calculate gradient
        grad_old = grad;
        grad = problem.full_grad(w);   
        
        
        if strcmp(options.sub_mode, 'PRELIM')
            % updata residual 
            r = grad;
            
            % updata beta        
            beta = r' * problem.A() * p / (p' * problem.A() * p);
            
            % update direction  
            p = - grad + beta * p;         
        
        elseif  strcmp(options.sub_mode, 'STANDARD')
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
        
        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        
        % store infos
        [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        

        % display infos
        if options.verbose
            fprintf('CG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, iter, f_val, gnorm, optgap);
        end
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', options.tol_gnorm);
    elseif optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);        
    elseif iter == options.max_epoch
        fprintf('Max iter reached: max_epoch = %g\n', options.max_epoch);
    end     
    
end
