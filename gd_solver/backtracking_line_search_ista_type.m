function L = backtracking_line_search_ista_type(problem, y_old, L0, eta) 
% This is a ISTA (FISTA) type backtracking algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       y_old       current point
%       L0          current stepsize
%       eta         constant (>1)
% Output:
%       L           new stepsize
%
% References:
%       Section 3 "ISTA with backtracking" algorithm in
%       Amir Beck and Marc Teboulle,
%       "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems,"
%       SIAM J. IMAGING SCIENCES, Vol. 2, No. 1, pp. 183-202.
%       
%                   
% This file is part of SGDLibrary.
%                   
% Created by H.Kasai on Nov. 19, 2018.
% 
% This code is ported the code written by Tiep Vu in
% https://github.com/tiepvupsu/FISTA/blob/master/fista_backtracking.m.


    L = L0;
    Lbar = L0;
    
    grad_y_old = problem.full_grad(y_old);
    
    while true
        
        % zk = problem.prox(y_old - 1/Lbar*grad, L0);
        %uk = problem.lincomb_vecvec(1, y_old, -1/Lbar, grads_y_old);
        uk = y_old - 1/Lbar * grad_y_old;
        %zk = problem.calculate_proximal_operator(uk, L0);
        zk = problem.prox(uk, 1/Lbar);
        
        
        F = problem.cost(zk);

        Q = calc_Q(problem, zk, y_old, grad_y_old, Lbar);

        if F <= Q 
            break;
        end
        
        Lbar = Lbar * eta; 
        L = Lbar; 
    end
    
end


%% computer Q 
function res = calc_Q(problem, x, y, grads_y, L) 

     res = problem.cost(y) + (x - y)'* grads_y + L/2*norm(x - y)^2 + problem.lambda * problem.reg(x);
    
    %cost_y = problem.calculate_cost(y);
    %x_y_diff = problem.lincomb_vecvec(1, x, -1, y);
    %x_y_diff_grad = problem.calculate_inner_product(x_y_diff, grads_y);
    %x_y_diff_snorm = problem.calculate_inner_product(x_y_diff, x_y_diff);
    %reg_val = problem.calculate_reg(x);
    
    %res = cost_y + x_y_diff_grad + L/2 * x_y_diff_snorm + problem.lambda * reg_val;
    
end 

