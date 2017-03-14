function [w, infos] = bb_sgd(problem, options)
% Big batch stochastic gradient descent algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
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
% Modified by H.Kasai on Jan. 12, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % extract options
    if ~isfield(options, 'step_init')
        step_init = 0.1;
    else
        step_init = options.step_init;
    end
    step = step_init;    
    
    if ~isfield(options, 'step_alg')
        step_alg = 'fix';
    else
        step_alg = options.step_alg;
    end  
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end        

    if ~isfield(options, 'batch_size')
        batch_size = 10;
    else
        batch_size = options.batch_size;
    end
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end      
    
    
    % initialize
    iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    K = batch_size;
    delta_K = floor(K/10);  
    prev_end_index = 0;
    Sample_end_reached = 0;
    K_max_reached = 0;     
    Sample_end = n * max_epoch;
    

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;    
    infos.cost = f_val;
    infos.gnorm = norm(problem.full_grad(w));        
    if store_w
        infos.w = w;       
    end
    infos.K = K;    
    
    % generate total random index
    total_samples = repmat(1:n, [1 max_epoch]);    
    perm_idx = randperm(Sample_end);
    perm_total_samples = total_samples(perm_idx);
    
    % set start time
    start_time = tic();
    
    % display infos
    if verbose > 0
        fprintf('BB SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end       
    
    % main loop
    while (optgap > tol_optgap) && ~Sample_end_reached

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
                    if block_size < batch_size
                        block_size = batch_size;
                    elseif block_size > 500
                        block_size = 500;
                    end

                    % calculate variance of gradient from block by block based on Welfordfs method
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
                    
                    % calculate variance of gradient by Welfordfs method
                    len_add = end_index - start_index + 1;
                    old_grad_ave = grad_ave;
                    grad_ave_rep = repmat(grad_ave, [1 len_add]);
                    grad_ave = grad_ave + sum((grad_cur - grad_ave_rep),2)/K;

                    grad_ave_rep = repmat(grad_ave, [1 len_add]);     
                    old_grad_ave_rep = repmat(old_grad_ave, [1 len_add]);    
                    grad_var = grad_var + sum(sum((grad_cur - grad_ave_rep) .* (grad_cur - old_grad_ave_rep)));
                    V_B = grad_var/(K-1);  
                end

                if grad_ave'*grad_ave >= V_B/K;
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
        if strcmp(step_alg, 'backtracking')        
            rho = 1/2;
            %c = 1e-4;
            c = 0.1;
            step = backtracking_line_search(problem, -grad, w, rho, c);
        elseif strcmp(step_alg, 'strong_wolfe')
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
        % update epoch
        epoch = epoch + 1;
        % calculate optimality gap
        f_val = problem.cost(w);
        optgap = f_val - f_opt;   
        % calculate norm of full gradient
        gnorm = norm(problem.full_grad(w));        

        % store infos
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm];         
        if store_w
            infos.w = [infos.w w];         
        end
        infos.K = [infos.K K];           

        % display infos
        if verbose > 0
            fprintf('BB SGD: Epoch = %03d, cost = %.16e, optgap = %.4e, K = %d\n', epoch, f_val, optgap, K);
        end

    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif Sample_end_reached
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end
    
end
