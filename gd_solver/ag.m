function [w, infos] = ag(problem, in_options)
% .
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2020
% Modified by H.Kasai on Oct. 28, 2020 (add optimal step (alpha) and beta)


    
    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = []; 
    local_options.algorithm                 = 'AG';    
    local_options.sub_mode                  = 'AG';
    local_options.use_fix_beta              = false;    
    local_options.use_optimal_alpha_beta    = false; 
    local_options.use_variable_beta         = false;
    local_options.use_restart               = 0;
    local_options.monotonic                 = 0;
    local_options.use_rada                  = 0;
    local_options.use_greedy                = 0;
    local_options.p                         = 1;
    local_options.q                         = 1;
    local_options.r                         = 4;
    local_options.beta                      = 0.01;

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     


    if ~problem.prox_flag
        options.sub_mode = 'AG';
    else
        options.sub_mode = 'APG';
    end
    
    if options.use_restart 
        if options.use_rada
            str = '(RADA)';
        elseif options.use_greedy
            str = '(Greedy)';                        
        else
            str = '(Restart)';
        end
    else
        if options.monotonic
            str = '(Monotonic)';
        else
            str = '';
        end
    end    
                
    disp_name = sprintf('%s %s', options.sub_mode, str);                
    
    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;
    w_old = w;
    prev_step = options.step_init;
    
    % initialize
    y = w;
    theta = 1;

    if options.use_greedy
        options.step_alg = 'no_change';
    end    
    
%     if strcmp(options.step_alg, 'ista_prox_backtracking')
%         prev_step = 1;
%     %elseif strcmp(options.step_alg, 'fix') && ~isempty(problem.prox) && isprop(problem, 'L')
%     elseif strcmp(options.step_alg, 'fix') && isprop(problem, 'L')
%         options.step_init = 1 / problem.L();
%     elseif strcmp(options.step_alg, 'fix')
%     end
    
    
    % for stepsize
    if strcmp(options.step_alg, 'ista_prox_backtracking')
        prev_step = 1;    
    elseif strcmp(options.step_alg, 'fix') || strcmp(options.step_alg, 'no_change')
        if isprop(problem, 'L')
            if problem.L > 0
                if isprop(problem, 'mu')
                    if problem.mu > 0 && options.use_fix_beta
                        % This casse is L-smooth and mu-strongly convex.
                        cn = problem.L/problem.mu;
                        if options.use_optimal_alpha_beta
                            options.step_init = 4/(3*problem.L + problem.mu);
                            if options.beta == 0.01
                                options.beta = (sqrt(3*cn+1)-2)/(sqrt(3*cn+1)+2); 
                            else % value by user
                                % use options.beta by user
                            end
                        else
                            options.step_init = 1/problem.L;
                            if options.beta == 0.01
                                options.beta = (sqrt(cn)-1)/(sqrt(cn)+1);   
                            else % value by user
                                % use options.beta by user
                            end                            
                        end                      
                    else
                        % This casse is L-smooth
                        options.step_init = 1/problem.L; 
                    end
                else
                    % This casse is L-smooth
                    options.step_init = 1/problem.L; 
                end
            else
                options.step_alg = 'backtracking';
            end
        end
    end    
    
    if strcmp(options.step_init_alg, 'bb_init')
        % initialize by BB step-size
        options.step_init = bb_init(problem, w);
    end    

    % for Rada-FISTA
    if options.use_rada
        ag_cnt  = 0;
        ag_flag = 1;
    end

    % greedy
    if options.use_greedy
        prev_step = 1.3 * options.step_init;
    end

    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    grad_old = [];
    fun_val_old = f_val;
    
    % display info
    if options.verbose
        fprintf('%s: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', disp_name, iter, f_val, gnorm, optgap);
    end  
    
    % set start time
    start_time = tic();    
    
    % main loop
    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)    
        
        % calculate stepsize        
        options.iter = iter;
        [step, ~] = options.linesearchfun(options.step_alg, problem, w, w_old, grad, grad_old, prev_step, options);  
        %[step, ~] = options.linesearchfun(options.step_alg, problem, y, y_old, grad, grad_old, prev_step, options); 
         
        %params_old = params_in;
        w_old = w;


        % calculate 
        w = y - step * grad;
        %w = problem.update_iterate(1, y, -1, step, [], grads, indice, options.alt_opt_mode, alt_opt_inner_iter);
        w_tmp = w;


        % proximal operator for APG
        if problem.prox_flag            
            w = problem.prox(w, step);
        end          

        % calculate nesterov step
        theta_new = 0.5 * (options.p + sqrt(options.q + options.r * theta^2)); % FISTA parameter
        beta = (theta - 1)/theta_new;
        
        % calculate 
        w_w_old_diff = w - w_old;
        if options.use_restart  % Restart, Rada, Greedy

            % calculate (y-w)
            y_w_diff = y-w;
            % calculate (y-w)'*(w-w_old)
            y_w_diff_w_w_old_diff_ip = y_w_diff' * w_w_old_diff;

            if y_w_diff_w_w_old_diff_ip > 0
                w = w_old;
                y = w;

                if options.use_rada % Rada-FISTA

                    ag_cnt = ag_cnt + 1;

                    if ag_cnt >= 4 % increase the value here if the condition number is big

                        if ag_flag
                            a = min(1, beta);
                            a_half = (4 + 1*a) / 5;
                            ag_xi = a_half^(1/30);

                            ag_flag = 0;
                        end

                        options.r = options.r * ag_xi;

                        if options.r < 3.999
                            t_lim = ( 2*options.p + sqrt( options.r * options.p^2 + (4-options.r)*options.q ) ) / (4 - options.r);
                            theta = max(2 * t_lim, theta);

                            fprintf('[%d] diff =  %.16e\n', iter, theta - 1);
                        end

                    else
                        % theta = 1;
                    end


                else
                    if options.use_greedy
                        % do nothing
                    else
                        theta = 1;
                    end
                end
            else
                % calculate y = w + (theta - 1)/theta_new * (w-w_old);
                %y = problem.lincomb_vecvec(1, w, (theta - 1)/theta_new, w_w_old_diff);
                a = min(1, beta);
                %y = problem.lincomb_vecvec(1, w, a, w_w_old_diff);
                y = w + a * w_w_old_diff;
            end
        else  % Baseline, Monotonic
            if ~options.monotonic
                if options.use_fix_beta
                    % Fix step (alpha) and beta
                    y = w + options.beta * w_w_old_diff; 
                elseif options.use_variable_beta
                    y = w + iter/(iter+3) * w_w_old_diff;                     
                else
                    % FISTA update!
                    % calculate y = w + (theta - 1)/theta_new * (w-w_old);
                    y = w + beta * w_w_old_diff; 
                end
            else
                fun_val = problem.cost(w);
                if iter > 1

                    if fun_val > fun_val_old
                        w = w_old;
                    else
                        % w = w;
                        % do nothing
                    end
                    
                end
                
                params_diff = w - w_old;
                %params_diff = problem.lincomb_vecvec(1, w, -1, params_old);
                %tmp = problem.lincomb_vecvec(theta/theta_new, params_diff, (theta - 1)/theta_new, w_w_old_diff);
                a = min(1, beta);
                %tmp = problem.lincomb_vecvec(theta/theta_new, params_diff, a, w_w_old_diff);
                tmp = theta/theta_new * params_diff + a * w_w_old_diff;
                %y = problem.lincomb_vecvec(1, w, 1, tmp);
                y = w + tmp;
            end

        end

        % Greedy
        if options.use_greedy
            %diff_tmp = problem.lincomb_vecvec(1, w_tmp, -1, params_old);
            diff_tmp = w_tmp - w_old;
            res = norm(diff_tmp);
            if iter == 0
                ag_res_init = res;
            end

            S = 1;
            if res > S * ag_res_init 
            %if 1
                xi = 0.96;

                step = max(options.step_init, step * xi); % To Do to handle containermap step

                %step_xi = problem.multiply_operator(xi, step);
                %step = problem.max_operator(options.step_init, step_xi);

                prev_step = step;
            end  
        end
        

        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        
        % store infos
        [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        
        fun_val_old = f_val;
        
        % print info
        if options.verbose
            fprintf('%s: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', disp_name, iter, f_val, gnorm, optgap);
        end      
        
        %fprintf('nor of w = %.16e\n', norm(w));        
        

        grad = problem.full_grad(y); 
        grad_old = grad; 
        theta = theta_new;         
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', options.tol_gnorm);
    elseif optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);        
    elseif iter == options.max_epoch
        fprintf('Max iter reached: max_epoch = %g\n', options.max_epoch);
    end  
    
end
