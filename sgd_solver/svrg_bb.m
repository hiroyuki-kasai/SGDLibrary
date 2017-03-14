function [w, infos] = svrg_bb(problem, options)
% Stochastic Variance gradient descent with Barzilai-Borwein (SVRG-BB) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       C. Tan, S. Ma, Y. Dai, Y. Qian, 
%       "Barzilai-Borwein Step Size for Stochastic Gradient Descent,"
%       NIPS, 2016.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Nov. 1, 2016
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
    num_of_bachces = floor(n / batch_size);        
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'max_inner_iter')
        max_inner_iter = num_of_bachces;
    else
        max_inner_iter = options.max_inner_iter;
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
    
    if ~isfield(options, 'permute_on')
        permute_on = 1;
    else
        permute_on = options.permute_on;
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

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    infos.gnorm = norm(problem.full_grad(w));        
    infos.cost = f_val;
    if store_w
        infos.w = w;       
    end    
    
    % set start time
    start_time = tic();
    
    % display infos
    if verbose > 0
        fprintf('SVRG BB: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end      

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % permute samples
        if permute_on
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

        for j=1:num_of_bachces
            
            % calculate variance reduced gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            
            % update w
            w = w - step * (full_grad + grad - grad_0);
            iter = iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * batch_size;        
        % update epoch
        epoch = epoch + 1;
        % calculate optgap
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

        % display infos
        if verbose > 0
            fprintf('SVRG BB: Epoch = %03d, cost = %.24e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end
    
end

