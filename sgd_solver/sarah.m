function [w, infos] = sarah(problem, in_options)
% StochAstic Recusive gRadient algoritHm (SARAH).
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takac, 
%       "SARAH: A novel method for machine learning problems using stochastic recursive gradient,"
%       ICML, 2017.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Sep. 29, 2017
% MOdified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.gamma = 1/8;
    local_options.sub_mode = '';
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    subinfos = [];     
    w = options.w_init;
    w_prev = w;
    num_of_bachces = floor(n / options.batch_size);  
    mode_plus_flag = 0;
    
    if ~isfield(options, 'max_inner_iter')
        options.max_inner_iter = num_of_bachces;
    end  
    
    if strcmp(options.sub_mode, 'Plus')    
        mode_plus_flag = 1;
    end
    
    % update step-size
    step = options.stepsizefun(total_iter, options);       

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0); 
    
    % display infos
    if options.verbose > 0
        fprintf('SARAH:%s Epoch = %03d, cost = %.24e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
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
        v0 = problem.full_grad(w);
        grad_calc_count = grad_calc_count + n;        
        
        % update w with full gradient
        w = w - step * v0;
        v = v0;
        
        if mode_plus_flag
            norm_v0 = norm(v0);
        end
        
        if options.store_subinfo
            [subinfos, f_val, optgap] = store_subinfos(problem, w, v, options, subinfos, epoch, total_iter, grad_calc_count, 0);           
        end         

        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);                 
         
            % calculate variance reduced gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            %grad_0 = problem.grad(w0, indice_j);
            grad_prev = problem.grad(w_prev, indice_j);
            
            % store variable
            w_prev = w;
            
            % update v
            v = grad - grad_prev + v;
            
            % update w
            w = w - step * v;
            
            % proximal operator
            if isfield(problem, 'prox')
                w = problem.prox(w, step);
            end  
        
            total_iter = total_iter + 1;
            
            % store sub infos
            if options.store_subinfo
                % measure elapsed time
                elapsed_time = toc(start_time);                
                [subinfos, f_val, optgap] = store_subinfos(problem, w, v, options, subinfos, epoch, total_iter, grad_calc_count, elapsed_time);           
            end
            
            
            if mode_plus_flag
                if norm(v) <= options.gamma * norm_v0
                    if options.verbose > 1
                        fprintf('\tSkipped at Inner iter = %04d\n', j);
                    end
                    break;
                end
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
            fprintf('SARAH:%s Epoch = %03d, cost = %.24e, optgap = %.4e, max_inner_iter = %d\n', options.sub_mode, epoch, f_val, optgap, j);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
    infos.subinfos = subinfos;
    
end

