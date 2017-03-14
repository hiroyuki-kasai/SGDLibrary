function [w, infos] = obfgs(problem, options)
% Online (limited-memory) quasi-newton methods (Online (L-)BFGS) algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       submode:    infinite memory:
%               N. N. Schraudolph, J. Yu and Simon Gunter, 
%               "A Stochastic Quasi-Newton Method for Online Convex Optimization," 
%               Int. Conf. Artificial Intelligence and Statistics (AIstats), pp.436-443, 
%               Journal of Machine Learning Research, 2007.
%
%               option.damped = true && option.regularized = true     
%                   X. Wang, S. Ma, D. Goldfarb and W. Liu, 
%                   "Stochastic Quasi-Newton Methods for Nonconvex Stochastic Optimization,"  
%                   arXiv preprint arXiv:1607.01231, 2016.
%
%               option.damped = false && regularized = true:    
%                  A. Mokhtari and A. Ribeiro, 
%                   "RES: Regularized Stochastic BFGS Algorithm," 
%                   IEEE Transactions on Signal Processing, vol. 62, no. 23, pp. 6089-6104, Dec., 2014.
%
%       submode:    limited memory
%               N. N. Schraudolph, J. Yu and Simon Gunter, 
%               "A Stochastic Quasi-Newton Method for Online Convex Optimization," 
%               Int. Conf. Artificial Intelligence and Statistics (AIstats), pp.436-443, 
%               Journal of Machine Learning Research, 2007.
%
%               A. Mokhtari and A. Ribeiro, 
%               "Global convergence of online limited memory BFGS," 
%               Journal of Machine Learning Research, 16, pp. 3151-3181, 2015.
%
%
% This file is part of SGDLibrary.
%                   
% Created by H.Kasai on Oct. 17, 2016
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
    num_of_bachces = floor(n / batch_size);        
    
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
    
    % 'Inf-mem' or 'Lim-mem'
    if ~isfield(options, 'sub_mode')
        sub_mode = 'Inf-mem';
    else
        sub_mode = options.sub_mode;    
        
        if ~isfield(options, 'mem_size')
            mem_size = 20;
        else
            mem_size = options.mem_size;
        end         
    end    
   
    % set delta for regularized oBFGS
    if ~isfield(options, 'regularized')
        delta = 0;
    else
        if ~options.regularized
            delta = 0;
        else
            if ~isfield(options, 'delta')
                delta = 0.1;
            else
                delta = options.delta; 
            end 
        end
    end 
    
     if ~isfield(options, 'damped')
        damped = false;
    else
        damped = options.damped;
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

    if strcmp(sub_mode, 'Lim-mem')
        s_array = [];
        y_array = [];            
    else
        % initialize BFGS matrix
        B = (delta>0)*delta*speye(d) + (delta==0)*speye(d);
    end    

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    infos.cost = f_val;
    infos.gnorm = norm(problem.full_grad(w));        
    if store_w
        infos.w = w;       
    end      
    
    % set start time
    start_time = tic();
    
    % display infos
    if verbose > 0
        if ~delta
            if ~damped
                fprintf('oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
            else
                fprintf('Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
            end
        else
            if ~damped
                fprintf('Reg-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
            else
                fprintf('Reg-Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
            end
        end
    end    

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % permute samples
        if permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end
        
        for j=1:num_of_bachces
            
            % update step-size
            if strcmp(step_alg, 'decay')
                step = step_init / (1 + step_init * lambda * total_iter);
            end                  
         
            % calculate gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w,indice_j);
            % store old iterate
            wo = w;            
            
            if strcmp(sub_mode, 'Lim-mem')
                % LBFGS two loop recursion
                HessGrad = lbfgs_two_loop_recursion(grad, s_array, y_array);
                w = w + step * HessGrad;    
            else
                % regularized Hessian and infinite memory (BFGS updating)
                w = w - step *( B\grad + delta*grad);
            end            
            
            % compute a stochastic gradient at the new point, same batch (double gradient evaluations)
            grad_new = problem.grad(w,indice_j);
            % update the curvature pairs
            s = w - wo;
            y = grad_new - grad - delta*s;
            
            if damped
                if strcmp(sub_mode, 'Lim-mem')
                    sty = s'*y;
                    HessGrad = lbfgs_two_loop_recursion(grad, s_array, y_array);
                    ytHessGrad = 0.2 * y' * HessGrad;
                    if(sty >= 0.2*ytHessGrad)
                        theta = 1;
                    else
                        theta = 0.8 * ytHessGrad / (ytHessGrad - sty);
                    end
                    r = theta * s + (1-theta) * HessGrad;
                else
                    sty = s'*y;
                    stBs = s'*B*s;
                    if(sty >= 0.2*stBs)
                        theta = 1;
                    else
                        theta = 0.8 * stBs / (stBs - sty);
                    end
                    % form r, convex combination of y and Bs
                    r = theta * y + (1-theta)*B*s;

                end
            else
                %y = s;
                r = y;
            end            

            if strcmp(sub_mode, 'Lim-mem')
                % store cavature pair
                % 'y' curvature pair is calculated from gradient differencing
                s_array = [s_array s];
                y_array = [y_array r]; 

                % remove overflowed pair
                if(size(s_array,2)>mem_size)
                    s_array(:,1) = [];
                    y_array(:,1) = [];
                end                
            else
                % update Hessian approximation
                B = B + (r*r')/(s'*r) - (B*s*s'*B)/(s'*B*s) + delta*speye(d);
            end           
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations (Dobly counted for grad and grad_new)
        grad_calc_count = grad_calc_count + 2* j * batch_size;        
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
            if ~delta
                if ~damped
                    fprintf('oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
                else
                    fprintf('Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
                end
            else
                if ~damped
                    fprintf('Reg-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
                else
                    fprintf('Reg-Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
                end
            end
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end        
end

