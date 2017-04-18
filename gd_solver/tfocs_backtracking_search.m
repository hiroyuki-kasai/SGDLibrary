function [ step ] = tfocs_backtracking_search(step_in, w, w_old, grad, grad_old, c1, c2)
%
% This file is part of GDLibrary.
%
% This file originally comes from https://github.com/bodono/apg.
% Modifeid by H.Kasai on Apr. 17, 2017

    s = w - w_old;
    y = grad_old - grad;
    
    step_hat = 0.5*(norm(s)^2)/abs(s'*y);
    step = min( c1*step_in, max( c2*step_in, step_hat ));  
end

