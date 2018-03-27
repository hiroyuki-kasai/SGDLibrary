function [w, infos] = cd_lasso_elasticnet(problem, options)
% Coordinate descent algorithm for LASSO problem.
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
% Created by H.Kasai on Apr. 18, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  
    A = problem.A();

    % extract options
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end      
    
    if ~isfield(options, 'tol_gnorm')
        tol_gnorm = 1.0e-12;
    else
        tol_gnorm = options.tol_gnorm;
    end    
    
    if ~isfield(options, 'max_iter')
        max_iter = 100;
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
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
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'lasso';
    else
        sub_mode = options.sub_mode;
    end     
    
    % initialise
    iter = 0;
    if strcmp(sub_mode, 'lasso')
        AtA = problem.AtA();
        squred_norm_col = diag(AtA);
    else
        AtA_l2 = problem.AtA_l2();
        squred_norm_col = diag(AtA_l2);
    end
    prox_th = ones(d, 1)./squred_norm_col;
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if ismethod(problem, 'reg')
        infos.reg = problem.reg(w);   
    end    
    if store_w
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();  
    
    % print info
    if verbose
        fprintf('CD (%s): Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
    end      

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        

        % update i-th coordinate
        if strcmp(sub_mode, 'lasso')
            for i = 1:d 
                w_except_i = w;
                w_except_i(i) = 0;
                residual = problem.residual(w_except_i);

                snc = squred_norm_col(i);
                w(i) = problem.prox(A(:, i)'*residual/snc, prox_th(i));
            end
        else
            for i = 1:d 
                w_except_i = w;
                w_except_i(i) = 0;
                residual = problem.residual(w_except_i, i);

                snc = squred_norm_col(i);
                w(i) = problem.prox(residual/snc, prox_th(i));
            end            
            
        end
        
        % calculate gradient
        grad = problem.full_grad(w);

        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap];        
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if ismethod(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end        
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
        if verbose
            fprintf('CD (%s): Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
    elseif optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);        
    elseif iter == max_iter
        fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end
