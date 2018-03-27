function [w, infos] = svrg(problem, in_options)
% Stochastic Variance gradient descent (SVRG) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Rie Johnson and Tong Zhang, 
%       "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction,"
%       NIPS, 2013.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options = [];
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    subinfos = [];      
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);  
    
    if ~isfield(options, 'max_inner_iter')
        options.max_inner_iter = num_of_bachces;
    end       

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0); 
    
    % display infos
    if options.verbose > 0
        fprintf('SVRG: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end      
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        % permute samples
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end

        % compute full gradient
        full_grad = problem.full_grad(w);
        % store w
        w0 = w;
        grad_calc_count = grad_calc_count + n;
        
        if options.store_subinfo
            [subinfos, f_val, optgap] = store_subinfos(problem, w, full_grad, options, subinfos, epoch, total_iter, grad_calc_count, 0);           
        end         

        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);                 
         
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            
            % update w
            v = full_grad + grad - grad_0;
            w = w - step * v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end  
        
            total_iter = total_iter + 1;
            
            % store sub infos
            if options.store_subinfo
                % measure elapsed time
                elapsed_time = toc(start_time);                
                [subinfos, f_val, optgap] = store_subinfos(problem, w, v, options, subinfos, epoch, total_iter, grad_calc_count, elapsed_time);           
            end            
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * options.batch_size;        
        epoch = epoch + 1;
         
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           

        % display infos
        if options.verbose > 0
            fprintf('SVRG: Epoch = %03d, cost = %.24e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
    infos.subinfos = subinfos;    
end

