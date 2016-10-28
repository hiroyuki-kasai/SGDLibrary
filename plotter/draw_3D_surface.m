function [ w_min, f_min, f_max ] = draw_3D_surface(cost, n, x_min, x_max, y_min, y_max)
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


    N = n;    
    xx = linspace(x_min, x_max, N);
    XX = repmat(xx, [N 1]);
    yy = linspace(y_min, y_max, N)';
    YY = repmat(yy, [1 N]);

    w = [XX(1,1); YY(1,1)];
    f_min = cost(w);
    f_max = cost(w);    
    
    for j=1:N
        for i=1:N
           %ZZ(i,j) = -2 * XX(i,j)^2 + XX(i,j) * YY(i,j)^2 + 4 * XX(i,j)^4;
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = cost(w);
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
    p = plot3(w_min(1),w_min(2), f_min, 'oy','MarkerFaceColor','y', 'MarkerSize', 6); hold off    
    xlabel('x(1)')
    ylabel('x(2)')    
    zlabel('cost')    
    alpha(.5)
    %legend('Optimal solution');
end

