function [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time)
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
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on ep. 25, 2017


    if ~epoch
        
        infos.iter = epoch;
        infos.time = 0;    
        infos.grad_calc_count = grad_calc_count;
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;
        infos.optgap = optgap;
        infos.gnorm = norm(problem.full_grad(w));    
        infos.cost = f_val;
        if ismethod(problem, 'reg')
            infos.reg = problem.reg(w);   
        end     
        if options.store_w
            infos.w = w;       
        end
        
    else
        
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        
        % calculate optimality gap
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;  
        % calculate norm of full gradient
        gnorm = norm(problem.full_grad(w));  
        
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if ismethod(problem, 'reg')            
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end          
        if options.store_w
            infos.w = [infos.w w];         
        end  
        
    end

end

