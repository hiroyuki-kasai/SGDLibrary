function [infos, f_val, optgap, grad, gnorm, subgrad, subgnorm, smooth_grad, smooth_gnorm] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time)
% Function to store statistic information
%
% Inputs:
%       problem         function (cost/grad/hess)
%       w               solution 
%       options         options
%       infos           struct to store statistic information
%       epoch           number of outer iteration
%       grad_calc_count number of calclations of gradients
%       elapsed_time    elapsed time from the begining
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%       grad            gradient
%       gnorm           norm of gradient
%       subgrad         subgradient
%       subgnorm        norm of subgradient
%       smooth_grad     smoothed gradient
%       smooth_gnorm    norm of smoothed gradient
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Sep. 25, 2017
% Modified by H.Kasai on Mar. 27, 2017
% Modified by H.Kasai on Oct. 30, 2020

    subgrad = [];
    subgnorm = [];

    if ~epoch
        
        infos.iter = epoch;
        infos.time = 0;    
        infos.grad_calc_count = grad_calc_count;
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;
        % calculate norm of full gradient
        grad = problem.full_grad(w);
        gnorm = norm(grad); 
        if ismethod(problem, 'full_subgrad')
            subgrad = problem.full_subgrad(w);
            subgnorm = norm(subgrad); 
            infos.subgnorm = subgnorm;  
        end   
        
        if ismethod(problem, 'full_smooth_grad')
            smooth_grad = problem.full_smooth_grad(w);
            smooth_gnorm = norm(smooth_grad); 
            infos.smooth_gnorm = smooth_gnorm;  
        end          
        
        infos.optgap = optgap;
        infos.best_optgap = optgap;
        infos.absoptgap = abs(optgap);        
        infos.gnorm = gnorm;    
        infos.cost = f_val;
        infos.best_cost = f_val;
        if ismethod(problem, 'reg')
            infos.reg = problem.reg(w);   
        end
        if options.store_w
            infos.w = w;       
        end
        if options.store_grad
            infos.grad = grad;       
        end        
        
    else
        
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % calculate optimality gap
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;  
        % calculate norm of full gradient
        grad = problem.full_grad(w);
        gnorm = norm(grad); 
        if ismethod(problem, 'full_subgrad')
            subgrad = problem.full_subgrad(w);
            subgnorm = norm(subgrad); 
            infos.subgnorm = [infos.subgnorm subgnorm];
        end 
        
        if ismethod(problem, 'full_smooth_grad')
            smooth_grad = problem.full_smooth_grad(w);
            smooth_gnorm = norm(smooth_grad); 
            infos.smooth_gnorm = [infos.smooth_gnorm smooth_gnorm];
        end          
        
        infos.optgap = [infos.optgap optgap];
        infos.absoptgap = [infos.absoptgap abs(optgap)];
        if optgap < infos.best_optgap(end)
            infos.best_optgap = [infos.best_optgap optgap];
        else
            infos.best_optgap = [infos.best_optgap infos.best_optgap(end)];
        end        
        infos.cost = [infos.cost f_val];
        if f_val < infos.best_cost(end)
            infos.best_cost = [infos.best_cost f_val];
        else
            infos.best_cost = [infos.best_cost infos.best_cost(end)];
        end
        infos.gnorm = [infos.gnorm gnorm]; 
        if ismethod(problem, 'reg')            
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end  
        if options.store_w
            infos.w = [infos.w w];         
        end
        if options.store_grad
            infos.grad = [infos.grad grad];       
        end         
        
    end

end

