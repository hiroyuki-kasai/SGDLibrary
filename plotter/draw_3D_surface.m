function [w_min, f_min, f_max] = draw_3D_surface(problem, n, x_min, x_max, y_min, y_max, octave_flag)
% Draw 3D surface.
%
% Inputs:
%       cost        problem cost function
%       n           grid samples
%       x_min       minimum value of x range  
%       x_max       maximum value of x range    
%       y_min       minimum value of y range    
%       y_max       maximum value of y range    
% Outputs:
%       w_min       w indicaing the minimum cost value
%       f_min       minimum cost value
%       f_max       maximum cost value
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 28, 2016
% Modified by H.Kasai on Mar. 27, 2018


    N = n;    
    xx = linspace(x_min, x_max, N);
    XX = repmat(xx, [N 1]);
    yy = linspace(y_min, y_max, N)';
    YY = repmat(yy, [1 N]);

    w = [XX(1,1); YY(1,1)];
    f_min = problem.cost(w);
    f_max = problem.cost(w);   
    
    ZZ = zeros(N, N);
    
    for j=1:N
        for i=1:N
           %ZZ(i,j) = -2 * XX(i,j)^2 + XX(i,j) * YY(i,j)^2 + 4 * XX(i,j)^4;
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = problem.cost(w);
           if ZZ(i,j) < f_min
               w_min = w;
               f_min = ZZ(i,j);
           end
           
           if ZZ(i,j) > f_max
               %w_max = w;
               f_max = ZZ(i,j);
           end           
        end
    end      

    
    %figure
    surf(XX,YY,ZZ); hold on
    plot3(w_min(1),w_min(2), f_min, 'oy','MarkerFaceColor','y', 'MarkerSize', 6); hold off    
    xlabel('x(1)')
    ylabel('x(2)')    
    zlabel('cost')   
    if ~octave_flag
        alpha(.5)
    end
    %legend('Optimal solution');
end

