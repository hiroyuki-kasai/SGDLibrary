function [w, infos] = subsamp_newton(problem, options)
% Sub-sampled Netwon method algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% Reference:
%       P. Xu, J. Yang, F. Ro-Khorasani, C. Re and M. W. Mahoney,
%       "Sub-sampled Newton Methods with Non-uniform Sampling,"
%       NIPS2016.
%
% This file is part of GDLibrary.
%
% Originally created by Peng Xu, Jiyan Yang on Feb. 20, 2016 (https://github.com/git-xp/Subsampled-Newton)
% Originally modified by H.Kasai on Mar. 16, 2017
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  


    % extract options
    if ~isfield(options, 'step_init')
        step_init = 1;
    else
        step_init = options.step_init;
    end
    step = step_init;
    
    if ~isfield(options, 'step_alg')
        step_alg = 'backtracking';
    else
        step_alg  = options.step_alg;
    end      
    
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
    
%     if ~isfield(options, 'step_init_alg')
%         % Do nothing
%     else
%         if strcmp(options.step_init_alg, 'bb_init')
%             % initialize by BB step-size
%             step_init = bb_init(problem, w);
%         end
%     end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'Uniform';
    else
        sub_mode = options.sub_mode;
    end

    if ~isfield(options, 'subsamp_hess_size')
        subsamp_hess_size = 200 * d;
    else
        subsamp_hess_size = options.subsamp_hess_size;
    end
    
    % Frequency of Hessian approximation for LS 
    if ~isfield(options, 'hess_update_freq')
        hess_update_freq = 10;
    else
        hess_update_freq = options.hess_update_freq;
    end 
    
    if ~isfield(options, 'r')
        % parameter tp compute approximate leverage scores
        r = min(10000, 20*d);
    else
        r = options.r;
    end     
    
    
    % initialise
    iter = 0;
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    % calculate gradient
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if ismethod(problem, 'reg')
        infos.reg = problem.reg(w);   
    end  
    
    if strcmp(sub_mode, 'RNS')
        rnorms = problem.x_norm();
    end    

    
    % set start time
    start_time = tic();

    if verbose
        fprintf('Subsampled Newton (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
    end       

    % main loop
    while (optgap > tol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)   
        
        if strcmp(sub_mode, 'Uniform')

            idx = randsample(n,subsamp_hess_size); % Sampling without replacement
            sub_square_hess_diag = problem.calc_square_hess_diag(w, idx);
            H = problem.diag_based_hess(w, idx, sub_square_hess_diag);
            
        elseif strcmp(sub_mode, 'RNS')

            square_hess_diag = problem.calc_square_hess_diag(w, 1:n);
            p = square_hess_diag.*rnorms';
            p = p/sum(p);              
            q = min(1, p*subsamp_hess_size);
            
            idx = rand(n,1)<q;
            p_sub = q(idx);
            H = problem.diag_based_hess(w, idx, square_hess_diag(idx)./p_sub);
            
        elseif strcmp(sub_mode, 'LS')
            
            % re-approximating scores every hess_update_freq iterations but never reweighting
            if mod(iter,hess_update_freq) == 0
                square_hess_diag = problem.calc_square_hess_diag(w, 1:n);
                lev = comp_apprx_ridge_lev(problem, r,sqrt(square_hess_diag));
                p0 = lev/sum(lev);
                q = min(1,p0*subsamp_hess_size);
            end
            
            idx = rand(n,1)<q; 
            p_sub = q(idx);
            sub_square_hess_diag = problem.calc_square_hess_diag(w, idx);
            H = problem.diag_based_hess(w, idx, sub_square_hess_diag./p_sub);
        end

        % calculate -Hv 
        [d,~] = pcg(H, -grad, 1e-6, 1000); 
        
        % linesearch
        if strcmp(step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, d, w, rho, c);
        elseif strcmp(step_alg, 'tfocs_backtracking') 
            if iter > 0
                alpha = 1.05;
                beta = 0.5; 
                step = tfocs_backtracking_search(step, w, w_old, grad, grad_old, alpha, beta);
            else
                %step = step_init;
            end            
        end          
        
        % update
        w_old = w; 
        w = w + step * d;
        
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end          
        
        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt; 
        % calculate gradient
        grad_old = -d;
        grad = problem.full_grad(w);           
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
            fprintf('Subsampled Newton (%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', sub_mode, iter, f_val, gnorm, optgap);
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


% Originally created by Peng Xu, Jiyan Yang on Feb. 20, 2016 (https://github.com/git-xp/Subsampled-Newton)
% Modified by H.Kasai on Mar. 16, 2017
function lev = comp_apprx_ridge_lev(problem, r, D)
    % there might be a better implementation for the sparse transform
    if nargin < 3
        D = [];
    end
    
    x_in = problem.x();
    X = x_in';
    lambda = problem.lambda();
    
    [n,d] = size(X);
    rn1 = randi(r, [n,1]);
    rn2 = randi(r, [d,1]);
    if D
        S1 = sparse(rn1, 1:n, (randi(2,[n,1])*2-3).*D, r, n);
    else
        S1 = sparse(rn1, 1:n, randi(2,[n,1])*2-3, r, n);
    end
    S2 = sparse(rn2, 1:d, sqrt(lambda)*(randi(2,[d,1])*2-3), r, d);
    SDX = S1*X + S2;
    [~,R] = qr(SDX,0);
    invRG = R\(randn(d, floor(d/2))/sqrt(floor(d/2)));
    if D
        lev = D.^2.*sum((X*invRG).^2,2);
    else
        lev = sum((X*invRG).^2,2);
    end
end
