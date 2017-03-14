function [w, infos] = slbfgs(problem, options)
% Stochastic limited-memory quasi-newton methods (Stochastic L-BFGS) algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       sub_mode:   SQN:
%                   Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. 
%                   "A stochastic quasi-Newton method for large-scale optimization," 
%                   SIAM Journal on Optimization, 26(2), 1008-1031, 2016.
%
%       sub_mode:   SVRG-SQN:
%                   Philipp Moritz, Robert Nishihara, Michael I. Jordan,
%                   "A Linearly-Convergent Stochastic L-BFGS Algorithm," 
%                   Artificial Intelligence and Statistics (AISTATS), 2016.
%
%       sub_mode:   SVRG LBFGS:
%                   R. Kolte, M. Erdogdu and A. Ozgur, 
%                   "Accelerating SVRG via second-order information," 
%                   OPT2015, 2015.
%
%                   
% Created by H.Kasai on Oct. 15, 2016
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
        if strcmp(options.step_alg, 'decay')
            step_alg = 'decay';
        elseif strcmp(options.step_alg, 'fix')
            step_alg = 'fix';
        else
            step_alg = 'decay';
        end
    end     
    
    if ~isfield(options, 'lambda')
        lambda = 0.1;
    else
        lambda = options.lambda;
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
    
    if batch_size > n
        batch_size = n;
    end   
    num_of_bachces = floor(n / batch_size);        
    
    if ~isfield(options, 'batch_hess_size')
        batch_hess_size = 20 * batch_size;
    else
        batch_hess_size = options.batch_hess_size;
    end    

    if batch_hess_size > n
        batch_hess_size = n;
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
    
    % SQN or SVRG-SQN or SVRG-LBFGS
    if ~isfield(options, 'sub_mode')
        sub_mode = 'SQN';
    else
        sub_mode = options.sub_mode;
    end
    
    if strcmp(sub_mode, 'SQN') || strcmp(sub_mode, 'SVRG-SQN')
        if ~isfield(options, 'L')
            L = 20;
        else
            L = options.L;
        end   
    elseif strcmp(sub_mode, 'SVRG-LBFGS')
        L = Inf;
    end
        
    if ~isfield(options, 'mem_size')
        mem_size = 20;
    else
        mem_size = options.mem_size;
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
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    
    s_array = [];
    y_array = [];    
    u_old = w;
    u_new = zeros(d,1);    

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
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
    end     

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % permute samples
        if permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end

        if strcmp(sub_mode, 'SVRG-SQN') || strcmp(sub_mode, 'SVRG-LBFGS')
            % compute full gradient
            %full_grad_new = problem.grad(w,1:n);
            full_grad_new = problem.full_grad(w);
            % count gradient evaluations
            grad_calc_count = grad_calc_count + n; 
        end          

        if strcmp(sub_mode, 'SVRG-LBFGS')
            if epoch > 0            
                % store cavature pair
                s_array = [s_array w - w0];
                y_array = [y_array full_grad_new - full_grad]; 

                % remove overflowed pair
                if(size(s_array,2)>mem_size)
                    s_array(:,1) = [];
                    y_array(:,1) = [];
                end     
            end
        end

        if strcmp(sub_mode, 'SVRG-SQN') || strcmp(sub_mode, 'SVRG-LBFGS')
            % store w for SVRG
            w0 = w;
            full_grad = full_grad_new;
        end          
      
        
        for j=1:num_of_bachces
            
            % update step-size
            if strcmp(step_alg, 'decay')
                step = step_init / (1 + step_init * lambda * total_iter);
            end                  
         
            % calculate gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            
            % calculate variance reduced gradient
            if strcmp(sub_mode, 'SVRG-SQN') || strcmp(sub_mode, 'SVRG-LBFGS')
                grad_w0 = problem.grad(w0,indice_j);
                grad = full_grad + grad - grad_w0;    
            end 
            
            if epoch > 0              
                % perform LBFGS two loop recursion
                Hg = lbfgs_two_loop_recursion(grad, s_array, y_array);
                % update w            
                w = w + (step*Hg);  
            else
                w = w - (step*grad); 
            end
            
            % calculate averaged w
            u_new = u_new + w/L;

            % update LBFGS vectors Hessian at every L iteration for 'SQN' or 'SVRG-SQN'
            % 'SVRG-LBFGS' does nothing because of L = Inf
            if(mod(total_iter,L)==0 && total_iter)                 
                
                % calcluate Hessian-vector product using subsamples
                sub_indices = datasample((1:n),batch_hess_size);
                % calculate hessian
                %H = problem.hess(w, sub_indices);
                %Hv = H*(u_new - u_old);
                % calculate hessian-vector product
                Hv = problem.hess_vec(w, u_new-u_old, sub_indices);

                % store cavature pair
                % 'y' curvature pair is calculated from a Hessian-vector product.
                s_array = [s_array u_new - u_old];
                y_array = [y_array Hv];                 
                
                % remove overflowed pair
                if(size(s_array,2)>mem_size)
                    s_array(:,1) = [];
                    y_array(:,1) = [];
                end                

                u_old = u_new;
                u_new = zeros(d,1);
                
                % count gradient evaluations
                grad_calc_count = grad_calc_count + batch_hess_size;                
            end            
            
            total_iter = total_iter + 1;
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
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end      
end

