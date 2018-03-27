function [w, infos] = svrg_bb(problem, in_options)
% Stochastic Variance gradient descent with Barzilai-Borwein (SVRG-BB) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       C. Tan, S. Ma, Y. Dai, and Y. Qian, 
%       "Barzilai-Borwein Step Size for Stochastic Gradient Descent,"
%       NIPS, 2016.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Nov. 1, 2016
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
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);  
    
    if ~isfield(options, 'max_inner_iter')
        options.max_inner_iter = num_of_bachces;
    end       
    step = options.step_init;

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('SVRG BB: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end      

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        % permute samples
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end
        
        if epoch 
            % store full gradient
            full_grad_old = full_grad;

            % compute full gradient
            full_grad = problem.grad(w,1:n);
  
            % automatic step size selection based on Barzilai-Borwein (BB)
            w_diff = w - w0;
            g_diff = full_grad - full_grad_old;
            step = 1/num_of_bachces * (w_diff' * w_diff) / (w_diff' * g_diff);   
            fprintf('step:%f\n', step);
        else
            % compute full gradient
            full_grad = problem.grad(w,1:n);
        end
        
        % store w
        w0 = w;
        grad_calc_count = grad_calc_count + n;        

        for j = 1 : num_of_bachces
            
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            
            % update w
            w = w - step * (full_grad + grad - grad_0);
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end  
            
            total_iter = total_iter + 1;
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
            fprintf('SVRG BB: Epoch = %03d, cost = %.24e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
end

