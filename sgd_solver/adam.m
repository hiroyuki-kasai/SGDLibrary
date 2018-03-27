function [w, infos] = adam(problem, in_options)
% Adam: A Method for stochastic optimization algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       sub_mode: Adam and AdaMax
%                   Diederik Kingma and Jimmy Ba,
%                   "Adam: A Method for Stochastic Optimization,"
%                   International Conference for Learning Representation (ICLR), 2015
%                   
% This file is part of SGDLibrary.
%                   
% Created by H.Kasai on Oct. 17, 2016
% Modified by H.Kasai on Mar. 25, 2018
% Note that partial code is originaly created by M.Pak (See https://github.com/mp4096/adawhatever)


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.sub_mode = 'Adam';    % "Adam" or "AdaMax"
    local_options.beta1 = 0.9;
    local_options.beta2 = 0.9;
    local_options.epsilon = 0.9;
    local_options.beta1 = 1e-8;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);     
    
    % initialise moment estimates
    m = zeros(d, 1);
    if strcmp(options.sub_mode, 'Adam')    
        v = zeros(d, 1);   
    else 
        u = zeros(d, 1);
    end

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);    
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('Adam-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
    end   

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
            step = options.stepsizefun(total_iter, options);                    
         
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w,indice_j);
            
            % increment total iteration numbers
            total_iter = total_iter + 1;
            
            % update biased first moment estimate
            m = options.beta1.*m + (1 - options.beta1).*grad;
            
            if strcmp(options.sub_mode, 'Adam')
                % update biased second raw moment estimate
                v = options.beta2.*v + (1 - options.beta2).*(grad.^2);
                % compute bias-corrected fist moment estimate
                m_hat = m./(1 - options.beta1^total_iter);
                % compute bias-corrected second raw moment estimate
                v_hat = v./(1 - options.beta2^total_iter);    
                
                % update w
                w = w - step * m_hat ./ (sqrt(v_hat) + options.epsilon);                
            else % 'AdaMax'
                % update the exponentially weighted infinity norm
                u = max(options.beta2.*u, abs(grad));       
                % compute the bias-corrected fist moment estimate
                m_hat = m./(1 - options.beta1^total_iter);  
                
                % update w
                w = w - step * m_hat ./ u;                   
            end
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
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
            fprintf('Adam-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end      
end

