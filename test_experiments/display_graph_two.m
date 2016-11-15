function [ ] = display_graph_two(x_category, y_category, algorithm_list, w_list, info_list)
% SHow graphs of optimizations
%
% Inputs:
%       y_category            "cost" or "optimality_gap"
%       algorithms_list     algorithms to be evaluated
%       w_list              solution produced by each algorithm
%       info_list           statistics produced by each algorithm
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2016

    
    % for plotting
    linetype = {'r:','b:','g:','b-','g:','g-','m:','m-','y:','y-'};    
    fontsize = 16;
    markersize = 5;
    linewidth = 2;    

    % initialize
    legend_str = cell(1,1);    
    alg_num = 0;

    % plot
    figure;
    for alg_idx=1:length(algorithm_list)
        if ~isempty(info_list{alg_idx})
            alg_num = alg_num + 1;  
            
            if strcmp(x_category, 'numofgrad')
                x_plot_data = info_list{alg_idx}.grad_calc_count;
            elseif strcmp(x_category, 'epoch')
                x_plot_data = info_list{alg_idx}.epoch;    
            else
            end
            
            
            if strcmp(y_category, 'cost')
                y_plot_data = info_list{alg_idx}.cost;
            elseif strcmp(y_category, 'optimality_gap')
                y_plot_data = info_list{alg_idx}.optgap;
            elseif strcmp(y_category, 'gnorm')
                y_plot_data = info_list{alg_idx}.gnorm;                
            end
            
            semilogy(x_plot_data, y_plot_data, linetype{alg_num}, 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
            
            legend_str{alg_num} = algorithm_list{alg_idx};
        else
            %
        end
    end
    hold off;

    % X label
    if strcmp(x_category, 'numofgrad');    
        xlabel('Number of gradient evaluations', 'FontSize', fontsize);
    elseif strcmp(x_category, 'epoch');
        xlabel('Epoch', 'FontSize', fontsize);    
    end    
    
    % Y label    
    if strcmp(y_category, 'cost');    
        ylabel('Cost', 'FontSize', fontsize);
    elseif strcmp(y_category, 'optimality_gap');
        ylabel('Optimality gap', 'FontSize', fontsize);
    elseif strcmp(y_category, 'gnorm');
        ylabel('Norm of gradient', 'FontSize', fontsize);        
    end
    legend(legend_str);
    set(gca, 'FontSize', fontsize);      
end

