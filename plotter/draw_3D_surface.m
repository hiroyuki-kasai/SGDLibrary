function [ w_min, f_min ] = draw_3D_surface(f, n, x_min, x_max, y_min, y_max)
    f_min = Inf;
    N = n;    
    xx = linspace(x_min, x_max, N);
    XX = repmat(xx, [N 1]);
    yy = linspace(y_min, y_max, N)';
    YY = repmat(yy, [1 N]);

    w = [XX(1,1); YY(1,1)];
    f_min = f(w);
    
    for j=1:N
        for i=1:N
           %ZZ(i,j) = -2 * XX(i,j)^2 + XX(i,j) * YY(i,j)^2 + 4 * XX(i,j)^4;
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = f(w);
           if ZZ(i,j) < f_min
               w_min = w;
               f_min = ZZ(i,j);
           end
        end
    end      

    
    figure
    surf(XX,YY,ZZ); hold on
    plot3(w_min(1),w_min(2), f_min, 'or','MarkerFaceColor','y'); hold off    
    xlabel('x(1)')
    ylabel('x(2)')    
    zlabel('f')    
    alpha(.4)
end

