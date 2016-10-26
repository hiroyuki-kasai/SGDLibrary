function [w, infos] = sag(problem, options)
% Stochastic average descent (SAG) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       N. L. Roux, M. Schmidt, and F. R. Bach, 
%       "A stochastic gradient method with an exponential convergence rate for finite training sets,"
%       NIPS, 2012.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Oct. 14, 2016


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
    
    if ~isfield(options, 'step_alg')
        step_alg = 'fix';
    else
        if strcmp(options.step_alg, 'decay')
            step_alg = 'decay';
        elseif strcmp(options.step_alg, 'fix')
            step_alg = 'fix';
        else
            step_alg = 'decay';
        end
    end    
    
    if ~isfield(options, 'lambda')
        lambda = 0.1;
    else
        lambda = options.lambda;
    end 
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end         

    if ~isfield(options, 'batch_size')
        batch_size = 1;
    else
        batch_size = options.batch_size;
    end
    num_of_bachces = floor(n / batch_size);    
    
    if ~isfield(options, 'max_epoch')
        max_epoch = inf;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'SAG';
    else
        sub_mode = options.sub_mode;
    end     
    
    if ~isfield(options, 'f_sol')
        f_sol = -Inf;
    else
        f_sol = options.f_sol;
    end     
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    
    % initialize
    iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    
    % prepare an array of gradients, and a valiable of average gradient
    grad_array = zeros(d, num_of_bachces);
    grad_ave = mean(grad_array, 2);

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_sol;
    infos.optgap = optgap;
    infos.cost = f_val;

    % set start time
    start_time = tic();
    
    % permute samples (ToDo)
    perm_idx = 1:n;     

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        for j=1:num_of_bachces
            
            % update step-size
            if strcmp(step_alg, 'decay')
                step = step_init / (1 + step_init * lambda * iter);
            end
            
            % calculate gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            
            % update average gradient
            if strcmp(sub_mode, 'SAG')
                grad_ave = grad_ave + (grad - grad_array(:, j)) / num_of_bachces;
            else % SAGA
                grad_ave = grad_ave + (grad - grad_array(:, j));                
            end
            % replace with new grad
            grad_array(:, j) = grad;  
            
            % update w
            w = w - step * grad_ave;
            iter = iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces * batch_size;        
        % update epoch
        epoch = epoch + 1;
        % calculate optgap
        f_val = problem.cost(w);
        optgap = f_val - f_sol;        

        % store infos
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];

        % display infos
        if verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end    
end
