function [w, infos] = sgd_cm(problem, in_options)
% Stochastic gradient descent (SGD) algorithm with classical momentum.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Ilya Sutskever, James Martens, George Dahl and Geoffrey Hinton, 	
%       "On the importance of initialization and momentum in deep learning,"
%       ICML, 2013.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Sep. 28, 2017
% Modified by H.Kasai on Mar. 25, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % set local options 
    local_options.sub_mode = 'CM';  % 'CM' or 'CM-NAG'
    local_options.mu = 0.99;
    local_options.epsilon = 1e-4;
    local_options.mu_max = 0.99;
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);  
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    v = zeros(d,1);
    num_of_bachces = floor(n / options.batch_size); 

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
        fprintf('SGD-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
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

        for j = 1 : num_of_bachces
            
            % update step-size
            %step = options.stepsizefun(total_iter, options);
            
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad =  problem.grad(w, indice_j);
            
            % update v
            if strcmp(options.sub_mode, 'CM')
                v = options.mu * v - options.epsilon * grad;
            else % 'CM-NAG'
                sup = -1 - log2(1+floor(total_iter/250));
                mu = min(1-pow2(sup), options.mu_max);
                v = mu * v - options.epsilon * grad;
            end

            % update w
            w = w + v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, 1);
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
            fprintf('SGD-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
        end

    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
end