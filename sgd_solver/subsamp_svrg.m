function [w, infos] = subsamp_svrg(problem, options)
% Sabsampled SVRG algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Subsampled SVRG:
%       R. Kolte, M. Erdogdu and A. Ozgur, 
%       "Accelerating SVRG via second-order information," 
%       OPT2015, 2015.
%
%                   
% Created by H.Kasai on Oct. 28, 2016
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
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'r')
        r = inf;
    else
        r = options.r;
    end   
    
    if r > d
        r = d;
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
    
    %
    sample_size = round(10*r*log(d));    
    
    % set start time
    start_time = tic();
    
    % display infos
    if verbose > 0
        fprintf('Sub-sampled SVRG: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end    

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % permute samples
        if permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end

       % compute full gradient
        full_grad = problem.grad(w,1:n);
        % store w
        w0 = w;
        grad_calc_count = grad_calc_count + n;        

        % calculated Hessian using subsamples every outer loop
        sub_indices = datasample((1:n), sample_size);   
        H = problem.hess(w, sub_indices);        
        [u, sigma,~] = svd(H);
        Q = u(:,1:r); 
        gamma = sigma(r+1,r+1);
        sigma = sigma(1:r,1:r);
        Sing_inv_gamma = diag(1./diag(sigma)) - diag(ones(r,1)/gamma);
        
        
        for j=1:num_of_bachces
            
            % update step-size
            if strcmp(step_alg, 'decay')
                step = step_init / (1 + step_init * lambda * iter);
            end                  
         
            % calculate variance reduced gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            grad_est = full_grad + grad - grad_0;  
            
            % update w
            v = -Q*(Sing_inv_gamma)*(Q' * grad_est) - (1/gamma)*grad_est;
            w = w + step * v;
                
            iter = iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * batch_size + sample_size;        
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
            fprintf('Sub-sampled SVRG: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end      
end

