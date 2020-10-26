function alpha = backtracking_line_search_prox(problem, w, alpha, rho) 
% This is a proximal backtracking algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       w      current point
%       alpha       current stepsize
%       rho         shrink constant (<1)
% Output:
%       alpha       new stepsize
%
% References:
%       
%                   
% This file is part of SGDLibrary.
%                   
% Created by H.Kasai on Nov. 19, 2018.
% 
%   F(x) := f(x) + g(x);
    
    
    w0 = w;
    
    %% f0
    % calculate f0
%     F0 = problem.calculate_cost(w0);
%     reg0 = problem.calculate_reg(w0);
%     g0 = problem.lambda * reg0;
%     f0 = F0 - g0;
    f0 = problem.differentiable_cost(w0);    
    
    % calculate g0
    grad0 = problem.full_grad(w0);
        
    % w = w - alpha * grads0
    w_out = w0 - alpha * grad0;
    
    % prox
    w_out = problem.prox(w_out, alpha); 
    
    
    %% f1
    fk = problem.differentiable_cost(w_out);
    
    diff = w_out - w0;
    
    while fk > f0 + grad0'*diff + 1/(2*alpha) * (diff'*diff)
        alpha = rho * alpha;
      
        % w = w - alpha * grads0
        w_out = w0 - alpha * grad0;
    
        % prox
        w_out = problem.prox(w_out, alpha); 
        
        %% fk
        fk = problem.differentiable_cost(w_out);
        diff = w_out - w0;
        
    end    
    
    
%     while fk >= f0 + grad_f(x)'*z + (0.5/step)*norm(z,2)^2
%         lambda = rho * step;
%         z = prox(x - lambda*grad_f(x),lambda*g);
%     end    
    
end

