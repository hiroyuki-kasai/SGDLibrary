function [w, infos] = sag(problem, in_options)
% Stochastic average descent (SAG) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
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
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.sub_mode = 'SAG';
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);     
    
    % prepare an array of gradients, and a valiable of average gradient
    grad_array = zeros(d, num_of_bachces);
    grad_ave = mean(grad_array, 2);

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);     
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
    end      

    % set start time
    start_time = tic();
    
    % permute samples (ToDo)
    perm_idx = 1:n;     

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);
            
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            
            % update average gradient
            if strcmp(options.sub_mode, 'SAG')
                grad_ave = grad_ave + (grad - grad_array(:, j)) / num_of_bachces;
            else % SAGA
                grad_ave = grad_ave + (grad - grad_array(:, j));                
            end
            % replace with new grad
            grad_array(:, j) = grad;  
            
            % update w
            w = w - step * grad_ave;
            
            % proximal operator
            if ismethod(problem, 'prox')                
                w = problem.prox(w, step);
            end  
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces * options.batch_size;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           

        % display infos
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end    
end
