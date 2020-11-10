function [w, infos] = ista(problem, in_options)
% (Fast) iterative soft (shrinkage)-thresholding algorithm ((F)ISTA).
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
% Created by H.Kasai on Apr. 18, 2017 for fista.m
% Modified by H.Kasai on Mar. 25, 2018
% Newly Created by H.Kasai on Oct. 23, 2020 for ista.m

    
    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = []; 
    local_options.algorithm = 'ISTA';    
    local_options.sub_mode  = 'FISTA';
    local_options.p         = 1;
    local_options.q         = 1;
    local_options.r         = 4;     

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     

    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;
    w_prev = w;
    y_prev = w;
    t_prev = 1;    

    % Lipschitz constant of the gradient of f
    if ~isfield(options, 'L')
        if isprop(problem, 'L')
            L = problem.L();
        else
            L = 1;
        end
    else
        L = options.L;
    end     
    Linv = 1/L;

    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    
    % display info
    if options.verbose
        fprintf('%s: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, iter, f_val, gnorm, optgap);
    end  
    
    % set start time
    start_time = tic();    
    
    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)    

        % calculate gradient
        if strcmp(options.sub_mode, 'FISTA')
            grad_y_old = problem.full_grad(y_prev);
            u = y_prev - Linv * grad_y_old;
        else
            u = w - Linv * grad;
        end
        w = problem.prox(u, Linv);
      

        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        
        % store infos
        [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        
        
        % print info
        if options.verbose
            fprintf('%s: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, iter, f_val, gnorm, optgap);
        end  
        
        if strcmp(options.sub_mode, 'FISTA')           
            % update paramters
            t = 0.5 * (options.p + sqrt(options.q + options.r * t_prev^2));            
            y = w + (t_prev - 1)/t * (w - w_prev);
        
            % store parameters
            w_prev = w;
            t_prev = t;
            y_prev = y;           
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
