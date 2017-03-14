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
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Nov. 02, 2016


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
%     num_of_bachces = floor(n / batch_size);    
    
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
    
%     if ~isfield(options, 'permute_on')
%         permute_on = 1;
%     else
%         permute_on = options.permute_on;
%     end     
    
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
    if store_w
        infos.w = w;       
    end      
    
    % generate total random index
    total_samples = repmat(1:n, [1 max_epoch]);    
    perm_idx = randperm(n * max_epoch);
    perm_total_samples = total_samples(perm_idx);
    
    
    K = batch_size;
    delta_K = 5;
    prev_end_index = 0;
    reach_end = 0;
    k_reached = 0; 

    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && ~reach_end

        K_prev = K;
         if ~k_reached
             
             if 0
            find_flag = 0;
            while ~find_flag
                % calcuate gradient 
                start_index = prev_end_index + 1;
                end_index = start_index + K-1;

                if end_index > n * max_epoch
                    end_index = n * max_epoch;
                    reach_end = 1;
                    find_flag = 1;                    
                end

                indice_j = perm_total_samples(start_index:end_index);
                grad =  problem.grad(w, indice_j);  

                if ~reach_end
                    % calculate variance
                    %old code          
                    if 1
                        var = 0;
                        for i=1:K
                            c_index = start_index + i - 1;
                            c_indice = perm_total_samples(c_index);
                            c_grad =  problem.grad(w, c_indice);  
                            diff_grad = c_grad-grad;
                            var = var + diff_grad'*diff_grad;
                        end

                        V_B = 1/(K-1) * var;
                    else
                        grad_hor = repmat(grad, [1 K]); 
                        grad_ind = problem.ind_grad(w, indice_j); 
                        diff = grad_hor - grad_ind;
                        V_B = 1/(K-1) * sum(sum(diff.*diff));
                    end

                    gnorm_square = grad'*grad;
                    if gnorm_square >= V_B/K;
                        find_flag = 1;
                        %fprintf('K is kept: %d, norm(grad)^2: %.4e, V_B/K: %.4e\n', K, gnorm_square,V_B/K);
                    else
                        K = K + delta_K;
                        %K = floor(K * 1.1);
                        if K >= n
                            K = n;
                            k_reached = 1;
                            find_flag = 1;
                        end
                       % fprintf('K increaseing: %d, norm(grad): %.4e, V_B/K: %.4e\n', K, norm(grad)^2,V_B/K);
                    end
                end
                %fprintf('find_flag:%d, k_reached:%d\n', find_flag, k_reached);
                %fprintf('start_index: %d, end_index: %d\n', start_index, end_index);
            end
            
             else


            find_flag = 0;
            first = 1;
            clear grad_array;
            grad_ave = zeros(d,1);
            grad_var = 0;            
            while ~find_flag
                if first
                    % calcuate gradient 
                    start_index = prev_end_index + 1;
                    end_index = start_index + K - 1;

                    if end_index > n * max_epoch
                        end_index = n * max_epoch;
                        reach_end = 1;
                        find_flag = 1;                    
                    end

                    indice_j = perm_total_samples(start_index:end_index);
                    grad_cur =  problem.ind_grad(w, indice_j);  
                    grad_array = grad_cur;
                    
                    first = 0;
                else
                    
                    % calcuate gradient 
                    start_index = end_index + 1;
                    end_index = start_index + delta_K - 1;

                    if end_index > n * max_epoch
                        end_index = n * max_epoch;
                        reach_end = 1;
                        find_flag = 1;                    
                    end     
                    
                    indice_j = perm_total_samples(start_index:end_index);
                    grad_cur =  problem.ind_grad(w, indice_j);    
                    grad_array = [grad_array grad_cur];
                    
                end
                
                grad_ind = sum(grad_array,2)/K;
                grad_hor = repmat(grad_ind, [1 K]); 
                %grad_ind = problem.ind_grad(w, indice_j); 
                diff = grad_array - grad_hor;
                V_B_org = 1/(K-1) * sum(sum(diff.*diff));
                        
                %grad_ind = sum(grad_array,2)/K;                
                
                len_add = size(grad_cur,2);
                old_grad_ave = grad_ave;
                grad_ave_rep = repmat(grad_ave, [1 len_add]);
                grad_ave = grad_ave + sum((grad_cur - grad_ave_rep),2)/K;
                
                grad_ave_rep = repmat(grad_ave, [1 len_add]);     
                old_grad_ave_rep = repmat(old_grad_ave, [1 len_add]);                   
                grad_var = grad_var + trace((grad_cur - grad_ave_rep)' * (grad_cur - old_grad_ave_rep));
                V_B = grad_var/(K-1);
                
                %find_flag = 1;
                kkk = 1;
                
                fprintf('VB = %.4e, VB_org  = %.4e\n', V_B, V_B_org);


                if ~reach_end
                    % calculate variance
                    %grad_hor = repmat(grad, [1 K]);
                    %diff = grad_hor - grad_array;
                    %V_B = 1/K * sum(sum(diff.*diff)); 
                    

                    if grad_ave'*grad_ave >= V_B/K;
                        find_flag = 1;
                        %fprintf('K is kept: %d, norm(grad): %.4e, V_B/K: %.4e\n', K, norm(grad)^2,V_B/K);
                    else
                        K = K + delta_K;
                        if K >= n
                            K = n;
                            k_reached = 1;
                            find_flag = 1;
                        end
                        %fprintf('K increaseing: %d, norm(grad): %.4e, V_B/K: %.4e\n', K, norm(grad)^2,V_B/K);
                    end
                end
            end
            
             end
            
        else
            start_index = prev_end_index + 1;
            end_index = start_index + K-1;

            if end_index > n * max_epoch
                end_index = n * max_epoch;
                reach_end = 1;
            end

            indice_j = perm_total_samples(start_index:end_index);
            grad =  problem.grad(w, indice_j);              
        end
        
        delta_K = floor(K/10);
        
        prev_end_index = end_index;
        
        %
        if strcmp(step_alg, 'backtracking')        
            rho = 1/2;
            %c = 1e-4;
            c = 0.1;
            step = backtracking_line_search(problem, -grad, w, rho, c);
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
        % calculate norm of gradient
        gnorm = norm(grad);        

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

        % display infos
        if verbose > 0
            fprintf('BB SGD: Epoch = %03d, cost = %.16e, optgap = %.4e, K = %d\n', epoch, f_val, optgap, K);
        end

    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    %elseif epoch == max_epoch
    elseif reach_end
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end
    
end
