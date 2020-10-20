function [w, infos] = sd(problem, in_options)
% Full steepest descent gradient algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Mar. 23, 2018
% Modified by H.Kasai on Oct. 20, 2020


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples(); 
    
    % set local options 
    local_options = [];    
    local_options.sub_mode = 'STANDARD';

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     

    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;
    if ~isfield(options, 'S')
        if strcmp(options.step_alg, 'exact')
            S = eye(d);
        end        
    else    
        S = options.S;
    end
    
    % initialize by BB step-size 
    if strcmp(options.step_init_alg, 'bb_init')
        options.step_init = bb_init(problem, w);
    end    
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    
    % display info
    if options.verbose
        if ~ismethod(problem, 'prox')
            fprintf('SD: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        else
            fprintf('PG: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
        end
    end  
    
    % set start time
    start_time = tic();      

    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)        

        % line search
        if strcmp(options.step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, -grad, w, rho, c);
        elseif strcmp(options.step_alg, 'exact')
            ls_options.sub_mode = sub_mode;
            ls_options.S = S;
            step = exact_line_search(problem, 'SD', -grad, [], [], w, ls_options);
        elseif strcmp(options.step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, -grad, w, c1, c2);
        elseif strcmp(options.step_alg, 'tfocs_backtracking') 
            if iter > 0
                alpha = 1.05;
                beta = 0.5; 
                step = tfocs_backtracking_search(step, w, w_old, grad, grad_old, alpha, beta);
            else
                step = options.step_init;
            end
        else
            step = options.step_init;
        end
        
        w_old = w;
        if strcmp(options.sub_mode, 'SCALING')
            % diagonal scaling 
            if isempty(S)
                h = problem.full_hess(w);
                S = diag(1./diag(h));
            end
            
            % update w
            w = w - step * S * grad;  
        else
            % update w
            w = w - step * grad;            
        end
        
        % proximal operator
        if ismethod(problem, 'prox')            
            w = problem.prox(w, step);
        end
        
        % store gradient
        grad_old = grad;

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
            if ~isfield(problem, 'prox')
                fprintf('SD: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
            else
                fprintf('PG: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
            end
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
