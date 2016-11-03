function [ alpha ] = backtracking_line_search(problem, p, x, rho, c)
% Backtracking line search
%
% Inputs:
%       problem     function (cost/grad/hess)
%       p           search direction
%       x           current iterate
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
% Created by H.Kasai on Feb. 29, 2016


    alpha = 1;
    f0 = problem.cost(x);
    g0 = problem.full_grad(x);
    x0 = x;
    x = x + alpha * p;
    fk = problem.cost(x);
    
    % repeat until the Armijo condition meets
    while fk > f0 + c * alpha * (g0'*p)
      alpha = rho * alpha;
      x = x0 + alpha * p;
      fk = problem.cost(x);
    end

end

