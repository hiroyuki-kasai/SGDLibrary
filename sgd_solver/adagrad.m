function [w, infos] = adagrad(problem, options)
% AdaGrad and RMSProp algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       sub_mode:   AdaGrad
%                   John Duchi, Elad Hazan and Yoram Singer, 
%                   gAdaptive Subgradient Methods for Online Learning and Stochastic Optimization,h 
%                   The Journal of Machine Learning Research 12: 2121-2159, 2011.
%       sub_mode:   RMSProp
%                   Tijmen Tieleman and G. Hinton,
%                   Lecture 6.5 - rmsprop, 
%                   COURSERA: Neural Networks for Machine Learning, 2012.
%       sub_mode:   AdaDelta
%                   M. D.Zeiler, 
%                   "AdaDelta: An Adaptive Learning Rate Method," 
%                   arXiv preprint arXiv:1212.5701, 2012.
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
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'AdaGrad';
    else
        sub_mode = options.sub_mode;
    end      
    
    if strcmp(sub_mode, 'RMSProp') || strcmp(sub_mode, 'AdaDelta')
        if ~isfield(options, 'beta')
            beta = 0.9;
        else
            beta = options.beta;
        end     
    end
    
    if ~isfield(options, 'epsilon')
        epsilon = 1e-8;
    else
        epsilon = options.epsilon;
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
    
    % initialise moment estimates
    if strcmp(sub_mode, 'AdaGrad')    
        gradHist = zeros(d, 1);  
    elseif strcmp(sub_mode, 'RMSProp')    
        r = zeros(d, 1);
    else % 'AdaDelta'
        r = zeros(d, 1);
        s = zeros(d, 1);        
    end   
    if store_w
        infos.w = w;       
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
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
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
            
            if strcmp(sub_mode, 'AdaGrad')
                % Update historical gradients
                gradHist = gradHist + grad.^2;
                
                % update w
                w = w - step * grad ./ (sqrt(gradHist) + epsilon);                  
            elseif strcmp(sub_mode, 'RMSProp')
                if ~total_iter
                    r = grad.^2;  
                else
                    % calculate accumulate squared gradient
                    r = beta * r + (1-beta)* grad.^2;                    
                end
                
               % update w
                w = w - step * grad ./ (sqrt(r) + epsilon);                   
            else % 'AdaDelta'
                if ~total_iter
                    r = grad.^2;
                else
                    % calculate accumulate squared gradient
                    r = beta * r + (1-beta)* grad.^2;                    
                end
                
                % update
                v = - (sqrt(s) + epsilon)./(sqrt(r) + epsilon) .* grad;
                % update accumulated updates (deltas)
                s = beta * s + (1-beta)* v.^2;

               % update w
                w = w + v;                   
            end
                
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * batch_size;        
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
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end    
end

