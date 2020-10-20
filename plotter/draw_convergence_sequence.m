function [  ] = draw_convergence_sequence(problem, opt_sol, algorithms_list, w_history, cost_history, varargin)
% Draw convergence sequence.
%
% Inputs:
%       problem             function (cost/grad/hess)
%       algorithms_list     algorithms to be evaluated
%       w_history           solution history produced by each algorithm    
% 
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 31, 2016


    % check the nunber of dimensions
    if problem.dim() > 2
        fprintf('Exit draw_convergence_animation because it only supports the case of dimention 2.\n');     
        return;
    end         

    % for plotting
    c_style =  {'ro-','bo-','mo-','go-','co-','yo-','r*-','b*-','m*-','g*-','c*-','y*-','r+--','b+--','m+--','g+--','c+--','y+--','rs:','bs:','ms:','gs:','cs:','ys:','r.','b.','c.','g.','m.','y.'};
  
    
    % calculate necessary # of algorithms
    alg_num = length(algorithms_list);  
    % calculate necessary # of rows
    row_num = ceil(length(algorithms_list)/3);
    % calculate plot ranges
    x_range_max = -Inf;
    x_range_min = Inf;
    y_range_max = -Inf;
    y_range_min = Inf; 
    
    
    if nargin < 6
        for alg_idx=1:alg_num
            w = w_history{alg_idx};

            max_cost = max(cost_history{alg_idx});

            if ~(any(isinf(w(:))) || any(isnan(w(:)))) && (max_cost < 10e8)
                if x_range_max <  max(w(1,:))
                    x_range_max = max(w(1,:));
                end   
                if x_range_min >  min(w(1,:))
                    x_range_min = min(w(1,:));
                end  

                 if y_range_max <  max(w(2,:))
                    y_range_max = max(w(2,:));
                end   
                if y_range_min >  min(w(2,:))
                    y_range_min = min(w(2,:));
                end  
            else

            end
        end        
    else
        
        range_input = varargin{1};

        x_range_min = range_input.x_min;
        x_range_max = range_input.x_max;
        y_range_min = range_input.y_min;
        y_range_max = range_input.y_max;        
    end   
            
    

    x_range = x_range_max - x_range_min;
    y_range = y_range_max - y_range_min;   
    x_range_max = x_range_max + x_range/4;
    x_range_min = x_range_min - x_range/4;    
    y_range_max = y_range_max + y_range/4;    
    y_range_min = y_range_min - y_range/4;  
    
    % generate mesh
    unit_len = floor(x_range_max - x_range_min)/50;
    if unit_len == 0
        unit_len = 1/50;
    end

    xCoarse = x_range_min:unit_len:x_range_max;
    yCoarse = y_range_min:unit_len:y_range_max;
    [XX,YY] = meshgrid(xCoarse,yCoarse); 
    row_size = size(XX,1);      
    col_size = size(XX,2);
    for j=1:col_size
        for i=1:row_size
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = problem.cost(w);
        end
    end
    
    
    %% plot
    figure;
    %suptitle('Convergence sequence'); 
    if alg_num < 3
        plot_col = alg_num;
    else
        plot_col = 3;
    end
    for alg_idx=1:alg_num
        subplot(row_num,plot_col,alg_idx); 
        
        w = w_history{alg_idx};
        
        max_cost = max(cost_history{alg_idx});        
        
        if ~(any(isinf(w(:))) || any(isnan(w(:)))) && (max_cost < 10e8)
            
            % draw contour
            contour(XX,YY,ZZ,50); hold on;
            
            % plot convergence sequence
            plot(w(1,:),w(2,:),c_style{alg_idx});hold on; 
            xlim([x_range_min, x_range_max]);
            ylim([y_range_min, y_range_max]);            
            title_str = sprintf('%s', algorithms_list{alg_idx});  
            
            % plot solution point
            if ~isempty(opt_sol)
                plot(opt_sol(1) ,opt_sol(2),'ko','MarkerSize',10);    % solution  
            end
            
            xlabel('x(1)')
            ylabel('x(2)')            
        
        else
             title_str = sprintf('%s (failed)', algorithms_list{alg_idx});             
        end

        hold off   
        
        % display title
        title(title_str);          
    end

end

