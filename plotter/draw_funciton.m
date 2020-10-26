function [] = draw_funciton(f, x_range_min, x_range_max, y_range_min, y_range_max, unit_len)

  
    xCoarse = x_range_min:unit_len:x_range_max;
    yCoarse = y_range_min:unit_len:y_range_max;
    [XX,YY] = meshgrid(xCoarse,yCoarse); 
    row_size = size(XX,1);      
    col_size = size(XX,2);
    for j=1:col_size
        for i=1:row_size
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = f(w);
        end
    end    
    
    figure
    
    % draw contour
    contour(XX,YY,ZZ,50) 
    
    % draw x-axis
    x = [x_range_min x_range_max];
    y = [0 0];
    line(x,y,'Color','black','LineStyle','-')
    
    % draw y-axis    
    x = [0 0];
    y = [y_range_min y_range_max];
    line(x,y,'Color','black','LineStyle','-')    

    
end

