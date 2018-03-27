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
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % extract options    
    if ~isfield(options, 'stepsizefun')
        options.stepsizefun = @stepsize_alg;
    else
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
    
    if r > d + 1
        r = d - 1;
    end
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end     
    
    if ~isfield(options, 'f_opt')
        options.f_opt = -Inf;
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
        options.store_w = false;
    end      
    
    
    % initialize
    total_iter = 0;    
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);     

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);    
    
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
        
        
        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);                
         
            % calculate variance reduced gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            grad_0 = problem.grad(w0, indice_j);
            grad_est = full_grad + grad - grad_0;  
            
            % update w
            v = -Q*(Sing_inv_gamma)*(Q' * grad_est) - (1/gamma)*grad_est;
            w = w + step * v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
                
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * batch_size + sample_size;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

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

