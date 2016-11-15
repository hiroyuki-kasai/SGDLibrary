function [ ] = display_graph_two_comparison(category, algorithm_list, w_list, info_list)
% SHow graphs of optimizations
%
% Inputs:
%       category            "cost" or "optimality_gap"
%       algorithms_list     algorithms to be evaluated
%       w_list              solution produced by each algorithm
%       info_list           statistics produced by each algorithm
% 
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2016

    
    % for plotting
    linetype = {'r:','r-','b:','b-','g:','g-','m:','m-','y:','y-'};
    fontsize = 16;
    markersize = 5;
    linewidth = 2;    

    % initialize
    legend_str = cell(1,1);    
    alg_num = 0;
    
    % Number of gradient evaluations v.s. {Cost, Optimality gap}
    figure;
    for alg_idx=1:length(algorithm_list)
        if ~isempty(info_list{alg_idx})
            alg_num = alg_num + 1;  
            if strcmp(category, 'cost');
                plot_data = info_list{alg_idx}.cost;
            elseif strcmp(category, 'optimality_gap');
                plot_data = info_list{alg_idx}.optgap;
            end
            semilogy(info_list{alg_idx}.grad_calc_count, plot_data, linetype{alg_num}, 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
            
            legend_str{alg_num} = algorithm_list{alg_idx};
        else
            %
        end
    end
    hold off;

    xlabel('Number of gradient evaluations', 'FontSize', fontsize);
    if strcmp(category, 'cost');    
        ylabel('Cost', 'FontSize', fontsize);
    elseif strcmp(category, 'optimality_gap');
        ylabel('Optimality gap', 'FontSize', fontsize);
    end
    legend(legend_str);
    set(gca, 'FontSize', fontsize);      
end

