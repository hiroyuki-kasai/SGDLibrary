function [w, infos] = bb_sgd(problem, in_options)
% Big batch stochastic gradient descent algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Soham De, Abhay Yadav, David Jacobs, Tom Goldstein, 
%       "Big Batch SGD: Automated Inference using Adaptive Batch Sizes,"
%       arXiv preprint, arXiv:1610.05792, 2016.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Nov. 07, 2016
% Modified by H.Kasai on Sep. 25, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  
    
    % set local options 
    local_options.step_alg = 'backtracking';
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     
    
    % initialize
    iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    K = options.batch_size;
    delta_K = floor(K/10);  
    prev_end_index = 0;
    Sample_end_reached = 0;
    K_max_reached = 0;     
    Sample_end = n * options.max_epoch;
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    infos.K = K;    
    
    % generate total random index
    total_samples = repmat(1:n, [1 options.max_epoch]);    
    perm_idx = randperm(Sample_end);
    perm_total_samples = total_samples(perm_idx);
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('BB SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end       
    
    % main loop
    while (optgap > options.tol_optgap) && ~Sample_end_reached

        if ~K_max_reached
            
            K_determined = 0;
            first = 1;
            grad_ave = zeros(d,1);
            grad_var = 0;    
            end_index = prev_end_index;
            sample_size = K;
            
            % Update K 
            while ~K_determined
                
                % determine sample index for gradient calculation
                start_index = prev_end_index + 1;
                end_index = start_index + sample_size - 1;
                
                if end_index > Sample_end
                    end_index = Sample_end;
                    Sample_end_reached = 1;
                    K_determined = 1;                    
                end  
                
                indices = perm_total_samples(start_index:end_index);                
                    
                if first

                    % calculate average gradient
                    grad_ave =  problem.grad(w, indices);    

                    block_size = K;
                    if block_size < options.batch_size
                        block_size = options.batch_size;
                    elseif block_size > 500
                        block_size = 500;
                    end

                    % calculate variance of gradient from block by block based on Welford?fs method
                    % to avoid big matrix calculation when K is large
                    for i=1:block_size:K
                        start_block_index = start_index + i - 1;
                        end_block_index = start_block_index + block_size - 1;
                        if end_block_index > start_index + K - 1 
                            end_block_index = start_index + K - 1;
                            block_size = end_block_index - start_block_index + 1;
                        end
                        
                        if end_block_index > Sample_end
                            end_block_index = Sample_end;
                            block_size = end_block_index - start_block_index + 1;
                            Sample_end_reached = 1;
                        end
                        
                        c_indices = perm_total_samples(start_block_index:end_block_index);
                        c_ind_grad =  problem.ind_grad(w, c_indices);                                  

                        old_grad_ave = grad_ave;
                        grad_ave_rep = repmat(grad_ave, [1 block_size]);
                        grad_ave = grad_ave + sum((c_ind_grad-grad_ave_rep),2)/end_block_index;

                        grad_ave_rep = repmat(grad_ave, [1 block_size]);
                        old_grad_ave_rep = repmat(old_grad_ave, [1 block_size]);
                        grad_var = grad_var + sum(sum((c_ind_grad-grad_ave_rep) .* (c_ind_grad-old_grad_ave_rep)));    
                        
                        if Sample_end_reached
                            break;
                        end
                    end
                    
                    V_B = grad_var/(K-1);

                    first = 0;
                else
                    
                    % calculate inidivisual gradient, where grad_cur is big matrix
                    grad_cur =  problem.ind_grad(w, indices);  
                    
                    % calculate variance of gradient by Welford?fs method
                    len_add = end_index - start_index + 1;
                    old_grad_ave = grad_ave;
                    grad_ave_rep = repmat(grad_ave, [1 len_add]);
                    grad_ave = grad_ave + sum((grad_cur - grad_ave_rep),2)/K;

                    grad_ave_rep = repmat(grad_ave, [1 len_add]);     
                    old_grad_ave_rep = repmat(old_grad_ave, [1 len_add]);    
                    grad_var = grad_var + sum(sum((grad_cur - grad_ave_rep) .* (grad_cur - old_grad_ave_rep)));
                    V_B = grad_var/(K-1);  
                end

                if grad_ave'*grad_ave >= V_B/K
                    K_determined = 1;
                else
                    K = K + delta_K;
                    sample_size = delta_K;
                    if K >= n
                        K = n;
                        K_max_reached = 1;
                        K_determined = 1;
                    end
                end

            end
            
            grad = grad_ave;
            
        else
            start_index = prev_end_index + 1;
            end_index = start_index + K - 1;

            if end_index > Sample_end
                end_index = Sample_end;
                Sample_end_reached = 1;
            end

            indices = perm_total_samples(start_index:end_index);
            grad =  problem.grad(w, indices);              
        end
        
        % Update delta_K
        delta_K = floor(K/10);
        prev_end_index = end_index;
        
        % do line search
        if strcmp(options.step_alg, 'backtracking')        
            rho = 1/2;
            %c = 1e-4;
            c = 0.1;
            step = backtracking_line_search(problem, -grad, w, rho, c);
        elseif strcmp(options.step_alg, 'strong_wolfe')
            c1 = 1e-4;
            c2 = 0.9;
            step = strong_wolfe_line_search(problem, -grad, w, c1, c2);            
        end

        % update w
        w = w - step * grad;
        iter = iter + 1;

        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        %grad_calc_count = grad_calc_count + num_of_bachces * batch_size;       
        grad_calc_count = prev_end_index;
        epoch = epoch + 1;
      
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);  
        infos.K = [infos.K K];           

        % display infos
        if options.verbose > 0
            fprintf('BB SGD: Epoch = %03d, cost = %.16e, optgap = %.4e, K = %d\n', epoch, f_val, optgap, K);
        end

    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif Sample_end_reached
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
end
