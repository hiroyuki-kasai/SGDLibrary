function [ HessGrad ] = lbfgs_two_loop_recursion( grad, s_array, y_array )
% Two loop recursion algorithm for L-BFGS.
%
% Reference:
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       Algorithm 7.4 in Section 7.2.
%    
% This file is part of SGDLibrary.
%
% Created H.Kasai on Oct. 17, 2016


    if(size(s_array,2)==0)
        HessGrad = -grad;
    else
        q = grad;

        for i = size(s_array,2):-1:1
            rk(i) = 1/(y_array(:,i)'*s_array(:,i));
            a(i) = rk(i)*s_array(:,i)'*q;
            q = q - a(i)*y_array(:,i);
        end

        Hk0 = (s_array(:,end)'*y_array(:,end))/(y_array(:,end)'*y_array(:,end));
        R = Hk0.*q;

        for jj = 1:size(s_array,2)
            beta = rk(jj)*y_array(:,jj)'*R;
            R = R + s_array(:,jj)*(a(jj) - beta);
        end

        HessGrad = -R; 
    end
end

