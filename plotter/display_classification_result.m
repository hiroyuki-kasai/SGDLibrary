function [ ] = display_classification_result(problem, algorithms_list, w_list, y_pred_list, accuracy_list, x_train, y_train, x_test, y_test)
% Display results of classification problem.
%
% Inputs:
%       problem             function (cost/grad/hess)
%       algorithms_list     algorithms to be evaluated
%       w_list              solution produced by each algorithm
%       y_pred_list         predicted results produced by each algorithm
%       accuracy_list       Accuracy values produced by each algorithm
%       x_train             train data x
%       y_train             train data y           
%       x_test              test data x           
%       y_test              test data y         
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2016


    % for plotting
    cmap = 'mbgrcy';
    cmap_len = length(cmap);  
    mmap = 'o+*xs';
    
    n_class = problem.classes();
    prob_name = problem.name();    
    d = problem.dim();    
    if strcmp(prob_name, 'logistic_regression')
        D = d + 1;
    elseif strcmp(prob_name, 'softmax_regression')
        D = d/n_class;
    else
        D = d;
    end
  

    %% display points
    figure
    % calculate necessary # of rows
    row_num = ceil(length(algorithms_list)/2) + 1;
    % calculate coodinate range
    offset = 0.2;
    x_max = max([x_train(1,:) x_test(1,:)]) + offset;
    x_min = min([x_train(1,:) x_test(1,:)]) - offset;    
    y_max = max([x_train(2,:) x_test(2,:)]) + offset;
    y_min = min([x_train(2,:) x_test(2,:)]) - offset; 
    if D > 3  
        z_max = max([x_train(3,:) x_test(3,:)]) + offset;
        z_min = min([x_train(3,:) x_test(3,:)]) - offset;         
    end
    
    % generate legend strings
    legend_str = cell(1); 
    for l = 1:n_class
        legend_str{l} = sprintf('Class %d', l);
    end  
    
    
    if D <= 3   
        % display train data and label
        subplot(row_num,2,1)            
        for l = 1:n_class
            % train
            subset_train = find(y_train == l);
            style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
            plot(x_train(1, subset_train), x_train(2, subset_train), style);  hold on
        end 
        hold off
        xlim([x_min x_max])
        ylim([y_min y_max])    
        legend(legend_str);    
        title('Train data') 

        % display test data and label    
        subplot(row_num,2,2)       
        for l = 1:n_class
            % test            
            subset_pred= find(y_test == l);
            style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
            plot(x_test(1, subset_pred), x_test(2, subset_pred), style);  hold on            
        end   
        hold off    
        xlim([x_min x_max])
        ylim([y_min y_max])     
        legend(legend_str);        
        title('Test data') 


        %% display predicted test data of each algorithm
        for alg_idx=1:length(algorithms_list)    
            w = w_list{alg_idx};
            y_pred = y_pred_list{alg_idx};

            % display predicted test data
            subplot(row_num,2,2+alg_idx)

            for l = 1:n_class
                subset_pred= find(y_pred == l);
                style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
                plot(x_test(1, subset_pred), x_test(2, subset_pred), style);  hold on               
            end          

            if strcmp(prob_name, 'linear_svm')
                % plot line of w(1)*x1 + w(2)*x2 + w(3)*0 + ....+ w(D-1)*0 + w(D) = 0
                x=x_min-1:x_max+1;
                y= -w(1)/w(2) * x - w(3)/w(2);
                plot(x,y, 'r','Linewidth', 2);    
                legend_str{l+1} = 'SVM';
            end
            hold off

            xlim([x_min x_max])
            ylim([y_min y_max])         
            title_str = sprintf('%s (Accuracy: %.3f)', algorithms_list{alg_idx}, accuracy_list{alg_idx});
            title(title_str)    
            legend(legend_str);
        end 
        
    else
        
        %legend_str ={'Train data','Solution', 'Prediction'};
        
        % display train data and label
        subplot(row_num,2,1)            
        for l = 1:n_class
            % train
            subset_train = find(y_train == l);
            style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
            plot3(x_train(1, subset_train), x_train(2, subset_train), x_train(3, subset_train), style);  hold on
        end 
        hold off
        xlim([x_min x_max])
        ylim([y_min y_max])    
        ylim([z_min z_max])  
        title('Train data')         
        
        xlabel('x(1)')
        ylabel('x(2)')
        zlabel('x(2)')
        grid on
            
        % display legend by adjusting its position
        ax=legend(legend_str);
        pos=get(ax,'position');
        pos(1)=pos(1)-0.001;
        set(ax,'position',pos)            
        % change view
        view(-39,10); 

        % display test data and label    
        subplot(row_num,2,2)       
        for l = 1:n_class
            % test            
            subset_pred= find(y_test == l);
            style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
            plot3(x_test(1, subset_pred), x_test(2, subset_pred), x_test(3, subset_pred),style);  hold on            
        end   
        hold off    
        xlim([x_min x_max])
        ylim([y_min y_max])  
        ylim([z_min z_max])         
        title('Test data') 
        
        xlabel('x(1)')
        ylabel('x(2)')
        zlabel('x(2)')
        grid on
            
        % display legend by adjusting its position
        ax=legend(legend_str);
        pos=get(ax,'position');
        pos(1)=pos(1)-0.001;
        set(ax,'position',pos)            
        % change view
        view(-39,10);      
        

        %% display predicted test data of each algorithm
        for alg_idx=1:length(algorithms_list)    
            w = w_list{alg_idx};
            y_pred = y_pred_list{alg_idx};

            % display predicted test data
            subplot(row_num,2,2+alg_idx)

            for l = 1:n_class
                subset_pred= find(y_pred == l);
                style = [cmap(mod(l - 1, cmap_len) + 1) mmap(floor((l - 1) / cmap_len) + 1)];
                plot3(x_test(1, subset_pred), x_test(2, subset_pred), x_test(3, subset_pred), style);  hold on                
            end          

%             if strcmp(prob_name, 'linear_svm')
%                 x=x_min:0.1:x_max;
%                 [X,Y] = meshgrid(x);
%                 Z= -w(1)/w(2) * X - w(2)*Y/w(3);
%                 surf(X,Y,Z)
%                 alpha(.1)
%                 legend_str{l+1} = 'SVM';
%             end
            hold off

            xlim([x_min x_max])
            ylim([y_min y_max])         
            ylim([z_min z_max])               
            title_str = sprintf('%s (Accuracy: %.3f)', algorithms_list{alg_idx}, accuracy_list{alg_idx});
            title(title_str)    
            
            xlabel('x(1)')
            ylabel('x(2)')
            zlabel('x(3)')
            grid on

            % display legend by adjusting its position
            ax=legend(legend_str);
            pos=get(ax,'position');
            pos(1)=pos(1)-0.001;
            set(ax,'position',pos)            
            
            % change view
            view(-39,10);
        end         
    end
    

end

