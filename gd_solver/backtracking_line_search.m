function alpha = backtracking_line_search(problem, dir, w, rho, c)
% Backtracking line search
%
% Inputs:
%       problem     function (cost/grad/hess)
%       dir         search direction
%       w           current iterate
%       rho         backtrack step between (0,1), e.g., 1/2
%       c           parameter between 0 and 1, e.g., 1e^-4
% Output:
%       alpha       step size calculated by this algorithm
%
% Reference:
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       Algorithm 3.1 in Section 3.1.
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Nov. 07, 2018


    alpha = 1;
    
    f0 = problem.cost(w);
    grad0 = problem.full_grad(w);
        
    w0 = w;
    w_out = w + alpha * dir;
    
    fk = problem.cost(w_out);
    
    % repeat until the Armijo condition meets
    %while fk > f0 - c * alpha * grads0_dirs
    while fk > f0 + c * alpha * grad0' * dir
        alpha = rho * alpha;
        w_out = w0 + alpha * dir;
        fk = problem.cost(w_out);
    end

end

