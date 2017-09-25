function [ ] = display_regression_result(problem, w_opt, algorithms_list, w_list, y_pred_list, mse_list, x_train, y_train, x_test, y_test)
% Display results of regression problem.
%
% Inputs:
%       problem             function (cost/grad/hess)
%       w_opt              solution
%       algorithms_list     algorithms to be evaluated
%       w_list              solution produced by each algorithm
%       y_pred_list         predicted results produced by each algorithm
%       mse_list            MSE values produced by each algorithm
%       x_train             train data x
%       y_train             train data y           
%       x_test              test data x           
%       y_test              test data y         
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2016


    d = problem.dim();

    %% display points
    figure
    % calculate necessary # of rows
    row_num = ceil(length(algorithms_list)/2);
    % calculate coodinate range
    offset = 0.2;
    x1_max = max(x_train(1,:)) + offset;
    x1_min = min(x_train(1,:)) - offset;   
    y_max = max(y_train) + offset;
    y_min = min(y_train) - offset;    
    
    if d == 2
        % display 2D plot when the dimension is 2 , i.e., (x1, y)
        
        % display prediction of each algorithm
        for alg_idx=1:length(algorithms_list)    
            
            subplot(row_num,2,alg_idx)
            
            % train data
            plot(x_train(1,:), y_train, 'go');  hold on     

            % solution line
            y_star_pred_line = w_opt' * x_test; 
            plot(x_test(1,:),  y_star_pred_line, 'bo'); hold on              

            % prediction line        
            y_pred_line = w_list{alg_idx}' * x_test;             
            plot(x_test(1,:),  y_pred_line, 'r+'); hold off                  

            % calculate cost 
            f_cost = problem.cost(w_list{alg_idx});
            
            xlim([x1_min x1_max])
            ylim([y_min y_max])         
            title_str = sprintf('%s (Train cost: %.3f, Test MSE: %.3f)', algorithms_list{alg_idx}, f_cost, mse_list{alg_idx});
            title(title_str) 
            
            xlabel('x')
            ylabel('y')
            
            % display legend
            legend('Train data','Solution', 'Prediction');         
        end 
    
    else
        % display 3D plot 
        % If the dimension is more than 4, i.e., (x1, x2, ..., xd, y),
        % onlhy (x1, x2, y) elements are displayed. 
        
        for alg_idx=1:length(algorithms_list)    

            subplot(row_num,2,alg_idx)
            
            % train data
            plot3(x_train(1,:),x_train(2,:), y_train, 'go'); hold on

            % solution line
            y_star_pred_line = w_opt' * x_test; 
            plot3(x_test(1,:), x_test(2,:), y_star_pred_line, 'bo'); hold on    

            % prediction line        
            y_pred_line = w_list{alg_idx}' * x_test;             
            plot3(x_test(1,:), x_test(2,:), y_pred_line ,'r+'); hold off  
            
            % calculate cost 
            f_cost = problem.cost(w_list{alg_idx});

            title_str = sprintf('%s (Train cost: %.3e, Test MSE: %.3e)', algorithms_list{alg_idx}, f_cost, mse_list{alg_idx});          
            title(title_str)    

            xlabel('x(1)')
            ylabel('x(2)')
            zlabel('y')
            grid on

            % display legend by adjusting its position
            ax=legend('Train data','Solution', 'Prediction');
            pos=get(ax,'position');
            pos(1)=pos(1)-0.01;
            set(ax,'position',pos)            
            
            % change view
            view(-39,10);   
        end
        
    end
end

