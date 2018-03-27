function [subinfos, f_val, optgap] = store_subinfos(problem, w, dir, options, subinfos, epoch, inner_iter, grad_calc_count, elapsed_time)
% Function to store sub statistic information
%
% Inputs:
%       problem         function (cost/grad/hess)
%       w               solution 
%       options         options
%       subinfos           struct to store statistic information
%       epoch           number of outer iteration
%       iter            number of inner iteration
%       grad_calc_count number of calclations of gradients
%       elapsed_time    elapsed time from the begining
% Output:
%       subinfos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Dec. 06, 2017


    if ~isempty(subinfos)
        
        subinfos.inner_iter = [subinfos.inner_iter inner_iter];
        subinfos.time = [subinfos.time elapsed_time];
        subinfos.grad_calc_count = [subinfos.grad_calc_count grad_calc_count];
        
        % calculate optimality gap
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;  
        % calculate norm of full gradient
        gnorm = norm(problem.full_grad(w));  
        
        subinfos.optgap = [subinfos.optgap optgap];
        subinfos.gnorm = [subinfos.gnorm gnorm];   
        subinfos.dnorm = [subinfos.dnorm norm(dir)];
        subinfos.cost = [subinfos.cost f_val];        
        

    else
        
        subinfos.inner_iter = inner_iter;
        subinfos.time = elapsed_time;
        subinfos.grad_calc_count = grad_calc_count;
        f_val = problem.cost(w);
        optgap = f_val - options.f_opt;
        subinfos.optgap = optgap;
        subinfos.gnorm = norm(problem.full_grad(w));  
        subinfos.dnorm = norm(dir);
        subinfos.cost = f_val;

        
    end

end

