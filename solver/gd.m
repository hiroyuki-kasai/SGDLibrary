function [w, infos] = gd(problem, options)
% Full gradient descent algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Oct. 25, 2016


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % extract options
    if ~isfield(options, 'step')
        step_init = 0.1;
    else
        step_init = options.step;
    end
    step = step_init;
    
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
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 1000;
    else
        max_epoch = options.max_epoch;
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
    
    if ~isfield(options, 'f_sol')
        f_sol = -Inf;
    else
        f_sol = options.f_sol;
    end    

    
    % initialise
    epoch = 0;
    indices = 1:problem.samples();    
    
    % store first infos
    clear infos;
    infos.epoch = epoch;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_sol;
    infos.optgap = optgap;
    grad = problem.grad(w, indices);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    
    % set start time
    start_time = tic();    

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (epoch < max_epoch)        

        % calculate gradient
        grad = problem.grad(w, indices);
        
        % update w
        w = w - step * grad;
        
        % update epoch        
        epoch = epoch + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_sol;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.epoch = [infos.epoch epoch];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count epoch*n];      
        infos.optgap = [infos.optgap optgap];        
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm];        
       
        % print info
        if verbose
            fprintf('GD: Epoch = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', epoch, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end    
    
end
