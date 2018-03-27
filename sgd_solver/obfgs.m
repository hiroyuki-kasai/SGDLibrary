function [w, infos] = obfgs(problem, in_options)
% Online (limited-memory) quasi-newton methods (Online (L-)BFGS) algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
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
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.sub_mode = 'Inf-mem';
    local_options.mem_size = 20;
    local_options.damped = false;
    local_options.delta = 0;
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);         
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);      

    if strcmp(options.sub_mode, 'Lim-mem')
        s_array = [];
        y_array = [];            
    else
        % initialize BFGS matrix
        B = (options.delta>0)*options.delta*speye(d) + (options.delta==0)*speye(d);
    end    

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);     
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        if ~options.delta
            if ~options.damped
                fprintf('oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
            else
                fprintf('Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
            end
        else
            if ~options.damped
                fprintf('Reg-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
            else
                fprintf('Reg-Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
            end
        end
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
            % store old iterate
            wo = w;            
            
            if strcmp(options.sub_mode, 'Lim-mem')
                % LBFGS two loop recursion
                HessGrad = lbfgs_two_loop_recursion(grad, s_array, y_array);
                w = w + step * HessGrad;    
            else
                % regularized Hessian and infinite memory (BFGS updating)
                w = w - step *( B\grad + options.delta*grad);
            end 
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
            
            % compute a stochastic gradient at the new point, same batch (double gradient evaluations)
            grad_new = problem.grad(w,indice_j);
            % update the curvature pairs
            s = w - wo;
            y = grad_new - grad - options.delta*s;
            
            if options.damped
                if strcmp(options.sub_mode, 'Lim-mem')
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

            if strcmp(options.sub_mode, 'Lim-mem')
                % store cavature pair
                % 'y' curvature pair is calculated from gradient differencing
                s_array = [s_array s];
                y_array = [y_array r]; 

                % remove overflowed pair
                if(size(s_array,2)>options.mem_size)
                    s_array(:,1) = [];
                    y_array(:,1) = [];
                end                
            else
                % update Hessian approximation
                B = B + (r*r')/(s'*r) - (B*s*s'*B)/(s'*B*s) + options.delta*speye(d);
            end           
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations (Dobly counted for grad and grad_new)
        grad_calc_count = grad_calc_count + 2* j * options.batch_size;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0
            if ~options.delta
                if ~options.damped
                    fprintf('oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
                else
                    fprintf('Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
                end
            else
                if ~options.damped
                    fprintf('Reg-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
                else
                    fprintf('Reg-Damped-oBFGS-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
                end
            end
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end        
end

