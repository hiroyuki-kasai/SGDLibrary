function [x] = soft_threshold(x0, tau)
    % returns argmin_x (tau||x||_1 + 1/2||x - x_0||) 
    
    len = size(x0, 1);
    x = x0;
    
    for i=1:len
        if x(i) < -tau
            x(i) = x(i) + tau;
        elseif x(i) > tau
            x(i) = x(i) - tau;
        else
            x(i) = 0;
        end
    end
end

