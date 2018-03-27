function [  ] = draw_convergence_animation(problem, algorithms_list, w_history, max_epoch, varargin)
% Draw convergence animation.
%
% Inputs:
%       problem             function (cost/grad/hess)
%       algorithms_list     algorithms to be evaluated
%       w_history           solution history produced by each algorithm    
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 28, 2016
% Modified by H.Kasai on Mar. 27, 2018


    octave_flag = is_octave();

    if nargin < 5
        speed = 0.5;
    else
        speed = varargin{1};
    end
    


    % check the nunber of dimensions
    if problem.dim() > 2
        fprintf('Exit draw_convergence_animation because it only supports the case of dimention 2.\n');     
        return;
    end         

    % for plotting
    c_style =  {'r','b','c','y','m','g','w','b','r','c','y','m','g','w'}; 
    fc_style = {'w','w','k','k','k','k','k','w','w','k','k','k','k','k'};
    
    % calculate necessary # of algorithms
    alg_num = length(algorithms_list);  
    % set the number of graphs per row
    if octave_flag
        max_row_num = alg_num;
    else
    max_row_num = 3;        
    end
    % calculate necessary # of rows
    row_num = ceil(length(algorithms_list)/max_row_num);
    % calculate plot ranges
    x_range_max = -Inf;
    x_range_min = Inf;
    y_range_max = -Inf;
    y_range_min = Inf;      
    for alg_idx=1:alg_num
        w = w_history{alg_idx};
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
    end
    x_range = x_range_max - x_range_min;
    y_range = y_range_max - y_range_min;   
    x_range_max = x_range_max + x_range/2;
    x_range_min = x_range_min - x_range/2;    
    y_range_max = y_range_max + y_range/2;    
    y_range_min = y_range_min - y_range/2;       

    
    %% plot
    figure('units','normalized','outerposition',[0 0 1 1]);
    
    % initialize
    N = 50;
    W_prev = cell(alg_num,1);
    len_array = zeros(alg_num,1);
    converge_flag_array = zeros(alg_num,1);    
    for alg_idx=1:alg_num
        
        subplot(row_num,max_row_num,alg_idx);
        [w_min, f_min, f_max] = draw_3D_surface(problem, N, ...
            x_range_min, x_range_max, y_range_min, y_range_max, octave_flag); hold on
        view(150,50);           
        %title_str = sprintf('%s ', algorithms_list{alg_idx});            
        %title(title_str);        

        converge_flag_array(alg_idx) = 0; 
        len_array(alg_idx) = size(w_history{alg_idx},2);
    end
    len_max = max(len_array);   
    
    pause(5)    

    % show plot
    for iter=1:len_max
        if iter > 1
            fprintf('epoch:%03d  ', iter);        
            for alg_idx=1:alg_num
                w = w_history{alg_idx};
                w_prev = W_prev{alg_idx};
                
                subplot(row_num,max_row_num,alg_idx);                
                
                if iter <= len_array(alg_idx)
                    % when not converged
                    f = problem.cost(w(:,iter));

                    % draw the line from the previous point to current point
                    line([w_prev(1) w(1,iter)],[w_prev(2) w(2,iter)],[w_prev(3) f], ...
                                                    'Marker', 'o', ...
                                                    'LineStyle','-', ...
                                                    'Color', c_style{alg_idx}, ...
                                                    'LineWidth',2); hold on
                    % store current point
                    W_prev{alg_idx} = [w(1,iter); w(2,iter); f];

                    fprintf('[%d] (%.2f,%.2f,%.2f)\t', alg_idx, w(1,iter), w(2,iter), f);  
                    
                    title_str = sprintf('%s [ iter: %03d ]', algorithms_list{alg_idx}, iter);            
                    title(title_str);  
                
                else
                    % when converged
                    if ~converge_flag_array(alg_idx)                        
                        converge_flag_array(alg_idx) = 1;
                        subplot(row_num,max_row_num,alg_idx); 
                        converge_str = sprintf('Converged at %d epoch !\nf=%.2e at (%.2f, %.2f)', ...
                                                iter, W_prev{alg_idx}(3), W_prev{alg_idx}(1), W_prev{alg_idx}(2));
                        hText = text(x_range_min, y_range_min, f_max - (f_max-f_min)/3, converge_str, ...
                                                    'BackgroundColor',c_style{alg_idx}, ...
                                                    'Color', fc_style{alg_idx}, ...
                                                    'Margin', 8, ...
                                                    'EdgeColor', 'w', ...
                                                    'LineWidth', 1.0, ...
                                                    'FontWeight', 'bold', ...
                                                    'FontSize', 14); hold on
                        % put text front
                        if ~octave_flag
                            set(hText, 'Layer', 'front');
                        end
                        fprintf('[%d] Converged\t\t', alg_idx);  
                        title_str = sprintf('%s [ iter: %03d ]', algorithms_list{alg_idx}, iter);            
                        title(title_str);                          
                    else
                        fprintf('[%d]\t\t\t\t', alg_idx);                              
                    end
                end
            end
            fprintf('\n');
        else
            for alg_idx=1:alg_num
                w = w_history{alg_idx};
                f = problem.cost(w(:,iter));
                W_prev{alg_idx} = [w(1,1); w(2,1); f];
            end
        end

        pause(speed)
        
    end

    % when reached maximum iteration
    for alg_idx=1:alg_num
        if ~converge_flag_array(alg_idx)
            subplot(row_num,max_row_num,alg_idx);   
            if iter == max_epoch+1
                converge_str = sprintf('Reached max %d epoch\nf=%.2e at (%.2f, %.2f)', ...
                                    iter-1, W_prev{alg_idx}(3), W_prev{alg_idx}(1), W_prev{alg_idx}(2));
            else
                converge_str = sprintf('Converged at %d epoch !\nf=%.2e at (%.2f, %.2f)', ...
                                    iter-1, W_prev{alg_idx}(3), W_prev{alg_idx}(1), W_prev{alg_idx}(2));                
            end
            hText = text(x_range_min,y_range_min, f_max - (f_max-f_min)/3, converge_str, ...
                                        'BackgroundColor',c_style{alg_idx}, ...
                                        'Color', fc_style{alg_idx}, ...
                                        'Margin', 8, ...       
                                        'EdgeColor', 'w', ...
                                        'LineWidth', 1.0, ...                                        
                                        'FontWeight', 'bold', ...
                                        'FontSize', 14); hold on  
            % put text front
            if ~octave_flag            
                set(hText, 'Layer', 'front');                                        
            end
        end
    end

    hold off         

end


 %% subfunction that checks if we are in octave
 function r = is_octave()
   persistent x;
   if (isempty(x))
     x = exist('OCTAVE_VERSION', 'builtin');
   end
   r = x;
 end

