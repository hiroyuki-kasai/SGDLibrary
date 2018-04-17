function [w, infos] = scr(problem, in_options)
% Minimize a continous, unconstrained function using the Adaptive Cubic Regularization method.
% 
% References
% ----------
% Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization. 
% Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295. Chicago 
% 
% Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.
% 
% Kohler, J. M., & Lucchi, A. (2017). Sub-sampled Cubic Regularization for Non-convex Optimization. arXiv preprint arXiv:1705.05933.
% 
% 
%
% Original Python code was created by J. M. Kohler and A. Lucchi (https://github.com/dalab/subsampled_cubic_regularization)
%
%
% This file is part of SGDLibrary.
%
% Ported to MATLAB code by K.Yoshikawa and H.Kasai on March, 2018.
% Modified by H.Kasai on Apr. 17, 2018

    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();   
    
    %% set local options 
    local_options.sampling_scheme = 'adaptive'; % 'linear', 'exponential'
    local_options.penalty_increase_multiplier = 2;  % multiply by..
    local_options.penalty_derease_multiplier = 2;  % divide by..
    local_options.initial_penalty_parameter = 0.01;
    local_options.successful_threshold = 0.1;
    local_options.very_successful_threshold = 0.9;
    local_options.grad_tol = 1e-9;

    % sampling
    local_options.Hessian_sampling = 1;
    local_options.gradient_sampling = 0;
    local_options.initial_sample_size_Hessian = 0.025;
    local_options.initial_sample_size_gradient = 0.25;
    local_options.unsuccessful_sample_scaling = 1.5;
    local_options.sample_scaling_Hessian = 1;
    local_options.sample_scaling_gradient = 1;

    % subproblem
    local_options.subproblem_solver = 'lanczos';  % alternatives: lanczos, cauchy_point, exact

    local_options.solve_each_i_th_krylov_space = 1;
    local_options.krylov_tol = 1e-1;
    local_options.exact_tol = 1e-2;
    local_options.keep_Q_matrix_in_memory = true;
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    sigma = options.initial_penalty_parameter;
    
    if options.verbose > 1    
        fprintf('- Subproblem_solver:%s\n', options.subproblem_solver)
        fprintf('- Hessian_sampling:%d\n', options.Hessian_sampling)
        fprintf('- Gradient_sampling:%d\n', options.gradient_sampling)
        fprintf('- Sampling_scheme:%s\n\n', options.sampling_scheme)
    end
    
    
    % decide mode
    if options.gradient_sampling == 0 && options.Hessian_sampling == 0 
        options.sampling_scheme = 'none';  
        mode = 'ARC';
    else
        mode = 'SCR';
    end
    
    % initialize
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;
    lambda_k = 0;
    successful_flag = false;

    grad = problem.full_grad(w);
    
    % compute exponential growth constant such that full sample size is reached in n_iterations
    if strcmp(options.sampling_scheme, 'exponential')
        exp_growth_constant = ((1-options.initial_sample_size_Hessian)*n)^(1/options.max_iter) ;
    end
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
        fprintf('%s:        : Epoch = %03d, cost = %.16e, optgap = %.4e\n', mode, iter, f_val, optgap);
    end    

    % set start time
    start_time = tic();
    
    for i = 0 : options.max_iter
        
        %% I: Subsampling         
        if strcmp(mode, 'SCR')
            
            % a) determine batchsize 
            if strcmp(options.sampling_scheme, 'exponential')
                
                sample_size_Hessian = options.Hessian_sampling*(fix(min([n, n*options.initial_sample_size_Hessian + exp_growth_constant^(i+1)]))+1) + (1-options.Hessian_sampling)*n;
                sample_size_gradient = options.gradient_sampling*(fix(min([n, n*options.initial_sample_size_gradient + exp_growth_constant^(i+1)]))+1) + (1-options.gradient_sampling)*n;
                
            elseif strcmp(options.sampling_scheme, 'linear')
                
                sample_size_Hessian = options.Hessian_sampling*fix(min([n, max([n*options.initial_sample_size_Hessian, n/options.max_iter*(i+1)])]))+(1-options.Hessian_sampling)*n;
                sample_size_gradient = options.gradient_sampling*fix(min([n, max([n*options.initial_sample_size_gradient, n/options.max_iter*(i+1)])]))+(1-options.gradient_sampling)*n;
                
            elseif strcmp(options.sampling_scheme, 'fix') % Added by HK
                
                sample_size_Hessian = options.Hessian_sampling*fix(min(n, n*options.initial_sample_size_Hessian))+(1-options.Hessian_sampling)*n;
                sample_size_gradient = options.gradient_sampling*fix(min(n, n*options.initial_sample_size_gradient))+(1-options.gradient_sampling)*n;
                
            elseif strcmp(options.sampling_scheme, 'adaptive')
                
                if i == 0
                    sample_size_Hessian = options.Hessian_sampling*fix(options.initial_sample_size_Hessian*n) + (1 - options.Hessian_sampling)*n;
                    sample_size_gradient = options.gradient_sampling*fix(options.initial_sample_size_gradient*n) + (1 - options.gradient_sampling)*n;
                else
                    % adjust sampling constant c such that the first step would have given a sample size of initial_sample_size
                    if i == 1
                        c_Hessian = (options.initial_sample_size_Hessian * n * sn.^2)/log(d);
                        c_gradient = (options.initial_sample_size_gradient * n * sn.^4)/log(d);
                    end
                    if successful_flag == false
                        sample_size_Hessian = options.Hessian_sampling*min([n,fix(sample_size_Hessian*options.unsuccessful_sample_scaling)]) + (1-options.Hessian_sampling)*n;
                        sample_size_gradient = options.gradient_sampling*min([n,fix(sample_size_gradient*options.unsuccessful_sample_scaling)]) +(1-options.gradient_sampling)*n;
                    else
                        sample_size_Hessian = options.Hessian_sampling*min(n,fix(max([(c_Hessian*log(d)/(sn.^2)*options.sample_scaling_Hessian),options.initial_sample_size_Hessian*n]))) + (1-options.Hessian_sampling)*n;            
                        sample_size_gradient = options.gradient_sampling*min(n,fix(max([(c_gradient*log(d)/(sn.^4)*options.sample_scaling_gradient),options.initial_sample_size_gradient*n]))) + (1-options.gradient_sampling)*n;                
                    end
                end
            end
        else % ARC mode
            sample_size_Hessian = n;
            sample_size_gradient = n;            
        end
        
        % b) draw batches
        if sample_size_Hessian < n
            int_idx_Hessian = randi([1, n], sample_size_Hessian,1); % KY
            bool_idx_Hessian = false(n,1);
            bool_idx_Hessian(int_idx_Hessian) = true;
            sub_hess_indices = find(bool_idx_Hessian);
        else
            sub_hess_indices = 1:n;
        end
        
        if sample_size_gradient < n
            int_idx_gradient = randi([1, n], sample_size_gradient,1);
            bool_idx_gradient = false(n,1);
            bool_idx_gradient(int_idx_gradient) = true;
            sub_grad_indices = find(bool_idx_gradient);
        else
            sub_grad_indices = 1:n;
        end
        
        n_samples_per_step = sample_size_Hessian + sample_size_gradient;
        
        
        %% II: Step computation
        % (Step.3) (a) recompute gradient either because of accepted step or because of re-sampling
        if options.gradient_sampling == 1 || successful_flag == 1
            %grad = gradient_f(w, new_X2, new_Y2, alpha);
            grad = problem.grad(w, sub_grad_indices');
            grad_norm = norm(grad);
            if grad_norm < options.grad_tol
                fprintf('Norm of gradient (%e) reached: grad_tol = %g\n', grad_norm, options.grad_tol);
                break
            end
        end
        
        % (Step.4) (b) call subproblem solver
        [s, lambda_k] = cr_subsolver(problem, w, grad, sub_hess_indices, sigma, successful_flag, lambda_k, options.subproblem_solver,...
                        options.exact_tol, options.krylov_tol, options.solve_each_i_th_krylov_space, options.keep_Q_matrix_in_memory);                    
                    
        sn = norm(s);
        
        %% III: Regularization Update
        % (Step.5) calcualte rho_k in Eq.(6)
        f_prev = problem.cost(w);
        f_curr = problem.cost(w+s);
        function_decrease = f_prev - f_curr; % f(w) - f(w+w)
        
        Hs = problem.hess_vec(w, s, sub_hess_indices);
        model_decrease = -(grad'*s + 0.5 * s'*Hs + (1/3)*sigma* (sn^3)); % f(w) - m(s) from Eq.(4)
       
        rho = function_decrease / model_decrease;
        if model_decrease >=0
            if options.verbose > 2
                fprintf('\tNegative model decrease (%e). This should not have happened\n', model_decrease);
            end
        end
        
        % (Step.6) update w if step s is successful in Eq.(7)
        if rho >= options.successful_threshold
            w = w + s;
            loss = f_curr;
            successful_flag = true;
            accstr = 'ACC';
        else
            loss = f_prev; 
            accstr = 'REJ';
        end

        % (Step.7) update penalty parameter in Eq.(8)
        if rho >= options.very_successful_threshold
            sigma = max([sigma/options.penalty_derease_multiplier, 1e-16]);
            % alternative (Cartis et al. 2011): sigma= max(min([grad_norm, sigma]), np.nextafter(0,1)) 
            trstr = 'CR+';
        elseif rho < options.successful_threshold
            sigma = options.penalty_increase_multiplier * sigma;
            successful_flag = false; 
            %fprintf('unscuccesful iteration\n');
            trstr = 'CR-';
        else
            trstr = '   ';            
        end
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n_samples_per_step;        
        iter = iter + 1;

        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        

        % display infos
        if options.verbose > 0
            fprintf('%s: %s %s: Epoch = %03d, cost = %.16e, optgap = %.4e', mode, accstr, trstr, iter, f_val, optgap);
            if options.verbose > 1
                fprintf(', sample (H,G)= (%d, %d)\n', sample_size_Hessian, sample_size_gradient)
            else
                fprintf('\n');
            end
        end 
        
        if optgap < options.tol_optgap
            fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
            break;
        end        
    end
    
    
    if iter == options.max_iter
        fprintf('Max iter reached: max_iter = %g\n', options.max_iter);
    end    
end


