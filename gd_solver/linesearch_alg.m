function [step, out_options] = linesearch_alg(step_alg, problem, w, w_old, grad, grad_old, prev_step, options)
% line-search (LS) algorithm.
%
% Inputs:
%       iter        number of iterations 
%       options     options
% Output:
%       step        stepsize
%       out_options options for output
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Nov. 07, 2018
% Modified by H.Kasai on Oct. 22, 2020


    ls_options = [];
    out_options = 0;
    
    % line search
    if strcmp(step_alg, 'backtracking')
        if isfield(options, 'backtracking_c')
            c = options.backtracking_c;
        else
            c = 1e-4;
        end
        
        if isfield(options, 'backtracking_rho')
            rho = options.backtracking_rho;
        else
            rho = 1/2;
        end

        step = backtracking_line_search(problem, -grad, w, rho, c);
    elseif strcmp(step_alg, 'exact')
        if isfield(options, 'sub_mode')
            ls_options.sub_mode = options.sub_mode;
        end
        if isfield(options, 'S')
            ls_options.S = options.S;
        end
        
        if strcmp(options.algorithm, 'CG')
            if strcmp(options.sub_mode, 'PRECON')
                step = exact_line_search(problem, options.algorithm, -grad, options.r_old, options.y_old, w, ls_options);
            else
                step = exact_line_search(problem, options.algorithm, -grad, options.r_old, [], w, ls_options);
            end
        else
            step = exact_line_search(problem, options.algorithm, -grad, [], [], w, ls_options);
        end
        
    elseif strcmp(step_alg, 'strong_wolfe')
        c1 = 1e-4;
        c2 = 0.9;
        grad_minus = - grad;
        step = strong_wolfe_line_search(problem, grad_minus, w, c1, c2);
    elseif strcmp(step_alg, 'tfocs_backtracking') 
        if ~isempty(grad_old)
            alpha = 1.05;
            beta = 0.5; 
            step = backtracking_search_tfocs(prev_step, w, w_old, grad, grad_old, alpha, beta);
        else
            step = options.step_init;
        end
    elseif strcmp(step_alg, 'prox_backtracking')
        rho = 1/2;
        step = backtracking_line_search_prox(problem, w, prev_step, rho);  
    elseif strcmp(step_alg, 'ista_prox_backtracking')
        eta = 1.5;
        %inv_prev_step = problem.inverse_scalar_operator(prev_step);
        inv_step = backtracking_line_search_ista_type(problem, w, 1/prev_step, eta);
        step = 1/inv_step;
    elseif contains(step_alg, 'decay')
        
        iter = options.iter;
        step_init = options.step_init;
        
        if strcmp(step_alg, 'decay-1')
            step = step_init / (1 + step_init * options.lambda * iter);
        elseif strcmp(step_alg, 'decay-2')
            step = step_init / (1 + iter);
        elseif strcmp(step_alg, 'decay-3')
            step = step_init / (options.lambda + iter);   
        elseif strcmp(step_alg, 'decay-4')
            step = step_init / sqrt(1 + step_init * options.lambda * iter);
        elseif strcmp(step_alg, 'decay-5')
            step = step_init / sqrt(1 + iter);
        elseif strcmp(step_alg, 'decay-6')
            step = step_init / sqrt(options.lambda + iter);  
        elseif strcmp(step_alg, 'decay-7')
            step = step_init / sqrt(2 + iter); 
        elseif strcmp(step_alg, 'decay-8')
            step = step_init / (3 + iter);            
        else
        end
        
    elseif contains(step_alg, 'gnorm') 
        
        iter = options.iter;        
        step_init = options.step_init;

        if strcmp(step_alg, 'gnorm-sub-1')  
            subgrad = problem.full_subgrad(w);
            gnorm = norm(subgrad);            
            step = step_init / gnorm;   
        elseif strcmp(step_alg, 'gnorm-sub-opt')  
            subgrad = problem.full_subgrad(w);
            gnorm = norm(subgrad);            
            step = (options.f_val - options.f_opt) / gnorm^2; 
        elseif strcmp(step_alg, 'gnorm-sub-est_opt')  
            subgrad = problem.full_subgrad(w);
            gnorm = norm(subgrad);            
            gamma = 10/(10+iter+1);
            step = (options.f_val - options.f_best + gamma) / gnorm^2;             
        elseif strcmp(step_alg, 'gnorm-dual-1')
            grad = problem.full_grad_dual(w);
            gnorm = norm(grad);            
            step = step_init / gnorm;        
        end 
        
    elseif contains(step_alg, 'exact_lasso') 
        
        step = problem.exact_line_search(w, -grad, options.idx, options.sign_flag);
        
    elseif strcmp(step_alg, 'no_change')

        step = prev_step;
        
    else
        % extract options
        if ~isfield(options, 'step_init')
            step = 0.1;
        else
            step = options.step_init;
        end
        
    end
    
  
end

