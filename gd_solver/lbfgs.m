function [w, infos] = lbfgs(problem, options)
% Limited-memory BFGS algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
%                   update_mode     'H'     Approximation inverse hessian (Eq. (6.17)) 
%                                   'B'     Approximation hessian (Eq. (6.19)) 
%                                   'Damp"  Damped BFGS updating by B (Procedure 18.2)
%
% Output:
%       w           solution of w
%       infos       information
%
% Reference:
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       Algorithm 7.5 in Section 7.2.
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Mar. 25, 2018


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
    
    if ~isfield(options, 'mem_size')
        mem_size = 20;
    else
        mem_size = options.mem_size;
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
    if ismethod(problem, 'reg')
        infos.reg = problem.reg(w);   
    end    
    if store_w
        infos.w = w;       
    end
    
    % set direction
    % Set the identity matrix to the initial inverse-Hessian-matrix
    % The first step is in the steepest descent direction
    InvHess = eye(d);
    p = - InvHess * grad;    
    
    % prepare array
    s_array = [];
    y_array = [];    
    
    % set start time
    start_time = tic();
    
    % print info
    if verbose
        fprintf('L-BFGS: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
    end     

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter) && ~stopping     
        
        % Revert to steepest descent if is not direction of descent                
        if (p'*grad > 0)
            p = -p;
        end 
        
        if iter > 0              
            % perform LBFGS two loop recursion
            p = lbfgs_two_loop_recursion(grad, s_array, y_array);
        end        
        
        % line search
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, p, w, rho, c);
        elseif strcmp(step_alg, 'exact')
            step = exact_line_search(problem, 'BFGS', p, [], [], w, []);
        elseif strcmp(step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, p, w, c1, c2);
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
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end        
        
        % calculate gradient     
        grad_old = grad; 
        grad = problem.full_grad(w);

        % store cavature pair
        % 'y' curvature pair is calculated from gradient differencing
        s = w - w_old;
        y = grad - grad_old;    
        s_array = [s_array s];
        y_array = [y_array y]; 

        % remove overflowed pair
        if(size(s_array,2) > mem_size)
            s_array(:,1) = [];
            y_array(:,1) = [];
        end          
        

        % calculate gradient
        grad = problem.full_grad(w);

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
            if ismethod(problem, 'reg')
                reg = problem.reg(w);
                infos.reg = [infos.reg reg];
            end            
            if store_w
                infos.w = [infos.w w];         
            end  
        else
            w_corrupted = true;
            w = w_old;
            stopping = true;            
        end
       
        % print info
        if verbose
            fprintf('L-BFGS: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
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
