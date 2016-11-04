function [alphas,fs,gs] = strong_wolfe_line_search(problem, d, x0, c1, c2)
%function [alphas,fs,gs] = strongwolfe(myFx,d,x0,fx0,gx0)
% Function strongwolfe performs Line search satisfying strong Wolfe conditions
% Input
%   myFx:   the optimized function handle
%   d:      the direction we want to search
%   x0:     vector of initial start
%   fx0:    the function value at x0
%   gx0:    the gradient value at x0  
% Output
%   alphas: step size
%   fs:     the function value at x0+alphas*d
%   gs:     the gradient value at x0+alphas*d
%
% Notice
%   I use f and g to save caculation. This funcion strongwolfe is called by LBFGS_opt.m.
% Ref
%   Numerical Optimization, by Nocedal and Wright
% Author:
%   Guipeng Li @THU
%   guipenglee@gmail.com
%
%
% This file is part of GDLibrary
%
% Modified by H.Kasai on November 1, 2016


    fx0 = problem.cost(x0);
    gx0 = problem.full_grad(x0);
    
    maxIter = 3;
    alpham = 20;
    alphap = 0;
    %c1 = 1e-4;
    %c2 = 0.9;
    alphax = 1;
    gx0 = gx0'*d;
    fxp = fx0;
    gxp = gx0;
    i=1;

    % Line search algorithm satisfying strong Wolfe conditions
    % Algorithms 3.2 on page 59 in Numerical Optimization, by Nocedal and Wright
    % alphap is alpha_{i-1}
    % alphax is alpha_i
    % alphas is what we want.
    while true
        xx = x0 + alphax*d;
        %[fxx,gxx] = feval(myFx,xx);
        fxx = problem.cost(xx);
        gxx = problem.full_grad(xx);      
        fs = fxx;
        gs = gxx;
        gxx = gxx'*d;
        if (fxx > fx0 + c1*alphax*gx0) || ((i > 1) && (fxx >= fxp))
            [alphas,fs,gs] = Zoom(problem,x0,d,alphap,alphax,fx0,gx0);
            return;
        end
        if abs(gxx) <= -c2*gx0,
            alphas = alphax;
            return;
        end
        if gxx >= 0,
            [alphas,fs,gs] = Zoom(problem,x0,d,alphax,alphap,fx0,gx0);
            return;
        end

        alphap = alphax;
        fxp = fxx;
        gxp = gxx;

        if i > maxIter
          alphas = alphax;
          return
        end  
        % r = rand(1);%randomly choose alphax from interval (alphap,alpham)
        r = 0.8;
        alphax = alphax + (alpham-alphax)*r;
        i = i+1;
    end
end% end of strongwolfe
    

%%%%%%%
function [alphas,fs,gs] = Zoom(problem,x0,d,alphal,alphah,fx0,gx0)
% Algorithms 3.2 on page 59 in 
% Numerical Optimization, by Nocedal and Wright
    % This function is called by strongwolfe
    c1 = 1e-4;
    c2 = 0.9;
    i =0;
    maxIter = 5;

    while true
        % bisection
        alphax = 0.5*(alphal+alphah);
        alphas = alphax;
        xx = x0 + alphax*d;
        %[fxx,gxx] = feval(myFx,xx);
        fxx = problem.cost(xx);
        gxx = problem.full_grad(xx);         
        fs = fxx;
        gs = gxx;
        gxx = gxx'*d;
        xl = x0 + alphal*d;
        %fxl = feval(myFx,xl);
        fxl = problem.cost(xl);

        if ((fxx > fx0 + c1*alphax*gx0) || (fxx >= fxl)),
            alphah = alphax;
        else
            if abs(gxx) <= -c2*gx0,
                alphas = alphax;
                return;
            end
            if gxx*(alphah-alphal) >= 0
                alphah = alphal;
            end
            alphal = alphax;
        end
        i = i+1;
        if i > maxIter
            alphas = alphax;
            return
        end
    end
end% end of Zoom


