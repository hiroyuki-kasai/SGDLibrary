function [w, infos] = iqn(problem, in_options)
% Incremental quasi-Newton method algorithm (IQN).
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
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
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.step_init = 1;
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);     

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init; 
    step = options.step_init;
    
    %x_0 = zeros(d,1);
    %rng('default')
    %x = x_0+1*normrnd(0,1e-5,[d,1]); 
    x = randn(d,1);
  
    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    
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
    if options.verbose > 0
        fprintf('IQN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end    

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        for j = 1 : n
            
            id = mod(j,n)+1;
            
            % calculate gradient
            grad =  problem.grad(w, id);
            
            % update w
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
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              

            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0
            fprintf('IQN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end

    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
    
end
