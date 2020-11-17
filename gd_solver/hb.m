function [w, infos] = hb(problem, in_options)
% Heavy Ball algorithm.
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
% Created by H.Kasai on Oct. 28, 2020


    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = [];
    local_options.step_alg = 'backtracking';
    local_options.beta = 0.01;    

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     

    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;
    w_old = w;
    prev_step = options.step_init;
    
    % for stepsize
    if strcmp(options.step_alg, 'fix') || strcmp(options.step_alg, 'no_change')
        if isprop(problem, 'L')
            if problem.L > 0
                if isprop(problem, 'mu')
                    if problem.mu > 0
                        % This casse is L-smooth and mu-strongly convex.
                        cn = problem.L/problem.mu;
                        options.step_init = 4/(sqrt(problem.L)+sqrt(problem.mu))^2;
                        options.beta = ( (sqrt(cn)-1)/(sqrt(cn)+1) )^2;                        
                    else
                        % This casse is L-smooth
                        options.step_init = 1/problem.L; 
                    end
                else
                    % This casse is L-smooth
                    options.step_init = 1/problem.L; 
                end
            else
                options.step_alg = 'backtracking';
            end
        end
    end

    % initialize by BB step-size 
    if strcmp(options.step_init_alg, 'bb_init')
        options.step_init = bb_init(problem, w);
    end    
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    grad_old = grad;
    
    % display info
    if options.verbose
        fprintf('HB: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
    end  
    
    % set start time
    start_time = tic();      

    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)  
        
        options.iter = iter;
        [step, ~] = options.linesearchfun(options.step_alg, problem, w, w_old, grad, grad_old, prev_step, options);   

        prev_step = step;
        w_w_old_diff = w - w_old;
        w_old = w;
  
        % update w
        w = w - step * grad + options.beta * w_w_old_diff;            

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
            fprintf('HB: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', iter, f_val, gnorm, optgap);
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
