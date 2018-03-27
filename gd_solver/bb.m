function [w, infos] = bb(problem, options)
% Barilai-Borwein descent algorithm.
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
% Created by H.Kasai on Oct. 31, 2016
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    p = problem.dim();
    n = problem.samples();  


    % extract options
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
        w = randn(p,1);
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
    if ismethod(problem, 'reg')
        infos.reg = problem.reg(w);   
    end  
    if store_w
        infos.w = w;       
    end
    
    % set direction
    p = - grad; 
    
    % set start time
    start_time = tic(); 
    
    % print info
    if verbose
        fprintf('BB: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
    end      

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        

        % Barzilai and Borwein Direction
%         if iter > 0
%             y = glad - grad_old;
%             s = w - w_old;
%             % caculate BB step size
%             step = (s'*s)./(s'*y);
%             if step <= 0
%                 p = - glad;
%             else
%                 p = - step * glad;
%             end
%         end
        if iter > 0
            y = glad - grad_old;
            s = w - w_old;
            % caculate BB step size            
            if rem(iter,2)
                step = (s'*s)./(s'*y);
            else
                step = (s'*y)./(y'*y);                
            end
            if step <= 0
                p = - glad;
            else
                p = - step * glad;
            end
            
            % store w and grad
            w_old = w;        
            grad_old = grad;     

            % update w
            w = w + p;
        
        else
            if 0
                % line search
                if strcmp(step_alg, 'backtracking')
                    rho = 1/2;
                    c = 1e-4;
                    step = backtracking_line_search(problem, p, w, rho, c);
                elseif strcmp(step_alg, 'exact')
                    ls_options.sub_mode = 'STANDARD';
                    step = exact_line_search(problem, 'SD', p, [], [], w, ls_options);
                else
                end    
            else
                step = 1;
            end
            
            % store w and grad
            w_old = w;        
            grad_old = grad;                 
       
            % update w
            w = w + step * p;            
        end
        
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end           

        % calculate gradient        
        glad = problem.full_grad(w); 
        
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
        if ismethod(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end  
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('BB: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
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
