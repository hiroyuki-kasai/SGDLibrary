function [w, infos] = ncg(problem, in_options)
% Nonlinear conjugate gradient algorithm.
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
% Created by H.Kasai on Oct. 30, 2016
% Modified by H.Kasai on Mar. 25, 2018
% Modified by H.Kasai on Oct. 22, 2020


    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = []; 
    local_options.algorithm = 'NCG';    
    local_options.beta_alg = 'FR';
    local_options.S = eye(d);

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);  


    % initialise
    iter = 0;
    stopping = false;
    w_corrupted = false;  
    grad_calc_count = 0;
    w = options.w_init; 
    w_old = w;    
    prev_step = options.step_init;     
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);

    % display infos
    if options.verbose
        fprintf('NCG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.beta_alg, iter, f_val, gnorm, optgap);
    end      
    
    % set first direction
    grad_old = grad;         
    d_old = -grad;
      
    
    % set start time
    start_time = tic();     

    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch) && ~stopping     
        
        % Revert to steepest descent if is not direction of descent                
        if (d_old'*grad_old > 0)
            d_old = -d_old;
        end        

        %line search
        [step, ~] = options.linesearchfun(options.step_alg, problem, w, w_old, -d_old, grad_old, prev_step, options);   
        prev_step = step;        
        
        % update w
        w_old = w;        
        w = w + step * options.S * d_old;
        
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end          
        
        % store old info   
        grad_old = grad;   
        % calculate gradient
        grad = problem.full_grad(w);        
        
        % updata beta
        switch options.beta_alg
            case 'FR'
                beta = (grad' * options.S * grad)/(grad_old' * options.S * grad_old);
            case 'PR'
                beta = ((grad - grad_old)'* options.S * grad)/(grad_old' * options.S * grad_old);
            otherwise
        end
        
        % avoid negative conjugate direction weights
        if beta < 0
            beta = max(0, beta);
        end        

        % update direction             
        d = - options.S * grad + beta * d_old;    
        
        % store d
        d_old = d;
        
        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;         
        
        % store info
        if ~(any(isinf(w(:))) || any(isnan(w(:)))) && ~isnan(f_val) && ~isinf(f_val)     
            [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);   
        else
            w_corrupted = true;
            w = infos.w(:,end);
            stopping = true;            
        end        
       
        % print info
        if options.verbose
            fprintf('NCG (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.beta_alg, iter, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', options.tol_gnorm);
    elseif optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);        
    elseif iter == options.max_epoch
        fprintf('Max iter reached: max_epoch = %g\n', options.max_epoch);
    elseif w_corrupted
        fprintf('Solution corrupted\n');        
    end    
    
end
