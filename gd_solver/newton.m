function [w, infos] = newton(problem, options)
% Netwon method algorithm.
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
%       sub_mode    'DAMPED'
%                   Amir Beck,
%                   "Introduction to Nonlinear Optimization Theory,
%                   Algorithms, and Applications with MATLAB,"
%                   MOS-SIAM Seris on Optimization, 2014.
%
%                   Algorithm in Section 5.2.
%
%       sub_mode    'CHOLESKY'
%                   Amir Beck,
%                   "Introduction to Nonlinear Optimization Theory,
%                   Algorithms, and Applications with MATLAB,"
%                   MOS-SIAM Seris on Optimization, 2014.
%
%                   Algorithm in Section 5.3.
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Oct. 25, 2016


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
    % calculate gradient
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    % calculate hessian
    hess = problem.full_hess(w);
    % calcualte direction    
    d = hess \ grad;
    if store_w
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();    

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        
        
        if strcmp(sub_mode, 'STANDARD')
            % update w
            w = w - d;            
        elseif strcmp(sub_mode, 'DAMPED') || strcmp(sub_mode, 'CHOLESKY') 
            
            if strcmp(step_alg, 'backtracking')
                rho = 1/2;
                c = 1e-4;
                step = backtracking_line_search(problem, -d, w, rho, c);
            else
                %
            end

            % update w
            w = w - step * d;            
        end

        % calculate gradient
        grad = problem.full_grad(w);
        % calculate hessian        
        hess = problem.full_hess(w);
        
        % calcualte direction        
        if strcmp(sub_mode, 'CHOLESKY') 
            [L, p] = chol(hess, 'lower');
            if p==0
                d = L' \ ( L \ grad);
            else
                d = grad;
            end
        else
            d = hess \ grad;        
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
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('Newton: Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
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
