function [ step_init ] = bb_init(problem, w)
%
% Barzilai-Borwein step-size initialization:
% 
%
% This file is part of GDLibrary.
%
% This file originally comes from https://github.com/bodono/apg.
% Modifeid by H.Kasai on Apr. 17, 2017

    grad = problem.full_grad(w);
    step = 1 / norm(grad);
    w_hat = w - step*grad;
    grad_hat = problem.full_grad(w_hat);
    
    s = w - w_hat;
    y = grad - grad_hat;
    step_init = abs(s'*y / (norm(y)^2));  
end

