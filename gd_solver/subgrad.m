function [w, infos] = subgrad(problem, in_options)
% Subgradient descent algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 20, 2020


    % set dimensions and samples
    d = problem.dim;
    n = problem.samples; 
    
    % set local options 
    local_options = [];    

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     

    % initialise    
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init; 
    w_old = w;
    prev_step = options.step_init;    

    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm, subgrad, subgnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    dir_old = grad;
    
    % display infos
    if options.verbose
        if ~problem.prox_flag 
            fprintf('Subgrad: Iter = %03d, cost = %.24e, gnorm = %.4e, subgnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, subgnorm, optgap);
        else
            fprintf('Prox-Subgrad: Iter = %03d, cost = %.24e, gnorm = %.4e, subgnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, subgnorm, optgap);
         end        
    end
    
    % set start time
    start_time = tic();      

    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)       
        
        dir = -(grad + subgrad); 

        options.iter = iter;
        options.subgrad = subgrad;  
        [step, ~] = options.linesearchfun(options.step_alg, problem, w, w_old, -dir, dir_old, prev_step, options);   

        prev_step = step;        
        
        % store previous w
        w_old = w;    
        
        % update w
        w = w + step * dir; 
        
        % proximal operator
        if problem.prox_flag
            w = problem.prox(w, step);
        end        
                
        % store gradient
        dir_old = dir;

        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        
        % store infos
        [infos, f_val, optgap, grad, gnorm, subgrad, subgnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        
       
        % print info
        if options.verbose
            if ~problem.prox_flag 
                fprintf('Subgrad: Iter = %03d, cost = %.24e, gnorm = %.4e, subgnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, subgnorm, optgap);
            else
                fprintf('Prox-Subgrad: Iter = %03d, cost = %.24e, gnorm = %.4e, subgnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, subgnorm, optgap);
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
