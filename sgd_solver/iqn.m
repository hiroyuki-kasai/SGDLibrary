function [w, infos] = iqn(problem, options)
% Incremental quasi-Newton method algorithm (IQN).
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       A. Mokhtari, M. Eisen, and A. Ribeiro, 
%       "An Incremental Quasi-Newton Method with a Local Superlinear Convergence Rate,"
%       ICASSP, 2017.
% 
% This file is part of SGDLibrary.
%
% Originally created by A. Mokhtari
% Modified by H.Kasai on Mar. 13, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % extract options
    if ~isfield(options, 'step_init')
        step = 1;
    else
        step = options.step_init;
    end
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
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
    
    %x_0 = zeros(d,1);
    rng('default')
    %x = x_0+1*normrnd(0,1e-5,[d,1]); 
    x = randn(d,1);
  

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
    
    
    % prepare arrays 
    t = zeros(d,n);
    y = zeros(d,n);    
    Q = zeros(d,d,n);
    
    % initialize arrays
    for i=1:n
        t(:,i) = x;
        y(:,i) = problem.grad(x,i);
        
        for j=1:d
            Q(j,j,i)=1;
        end
    end

    g = problem.full_grad(x);
    B = eye(d);
    u = x;   
    
    % display infos
    if verbose > 0
        fprintf('IQN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end    

    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        for j=1:n
            
            id = mod(j,n)+1;
            
            % calculate gradient
            grad =  problem.grad(w, id);
            
            if norm(w - t(:,id)) > 0
                s = w - t(:,id);
                yy = grad-y(:,id);

                stoc_Hessian = Q(:,:,id) + (yy*yy'/(yy'*s)) - Q(:,:,id)*s*s'*Q(:,:,id)/(s'*Q(:,:,id)*s); 

                B = B+(1/n)*(stoc_Hessian-Q(:,:,id));
                u = u+(1/n)*(stoc_Hessian*w- Q(:,:,id)*t(:,id)  );
                g = g+(1/n)*(grad-y(:,id));

                Q(:,:,id) = stoc_Hessian;
                y(:,id) = grad;
                t(:,id) = w;
                
                ww = (B^(-1))*(u-g);
                w = step * ww + (1-step) * w ;
    
            else
                w = w;
            end

            % update w
            iter = iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;        
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

        % display infos
        if verbose > 0
            fprintf('IQN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end

    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end
    
end
