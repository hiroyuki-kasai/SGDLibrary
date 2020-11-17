function [ ] = display_graph(x_category, y_category, algorithm_list, w_list, info_list, scale, line, width, xlim_range_in, ylim_range_in)
% SHow graphs of optimizations
%
% Inputs:
%       x_category          "numofgrad" or "iter" or "epoch" or "grad_calc_count"
%       y_category          "cost" or "optimality_gap" or "gnorm" or "subgnorm"
%       algorithms_list     algorithms to be evaluated
%       w_list              solution produced by each algorithm
%       info_list           statistics produced by each algorithm
% 
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Oct. 23, 2016
% Modified by H.Kasai on Apr. 24, 2017

    if nargin < 6
        scale_type = 'semilogy';
    else
        scale_type = scale;
    end
    
    if nargin < 7
        line_type = 'line';
    else
        line_type = line;
    end    
    
    if nargin < 8
        linewidth = 2;
    else
        linewidth = width;
    end 
    
    if nargin < 9
        xlim_range = [];
    else
        xlim_range = xlim_range_in;
    end 
    
    if nargin < 10
        ylim_range = [];
    else
        ylim_range = ylim_range_in;
    end     
    
    % for plotting
    if strcmp(line_type, 'line')
        linetype = {'r','b','m','g','c','y','r--','b--','c--','g--','m--','y--','r:','b:','c:','g:','m:','y:','r.','b.','c.','g.','m.','y.'};
    elseif strcmp(line_type, 'line-with-mark')
        linetype = {'ro-','bo-','mo-','go-','co-','yo-','r*-','b*-','m*-','g*-','c*-','y*-','r+--','b+--','m+--','g+--','c+--','y+--','rs:','bs:','ms:','gs:','cs:','ys:','r.','b.','c.','g.','m.','y.'};
    else
    end
    fontsize = 16;
    markersize = 5;


    % initialize
    legend_str = cell(1);    
    alg_num = 0;

    % plot
    figure;
    for alg_idx=1:length(algorithm_list)
        if ~isempty(info_list{alg_idx})
            alg_num = alg_num + 1;  
            
            if strcmp(x_category, 'numofgrad')
                x_plot_data = info_list{alg_idx}.grad_calc_count;
            elseif strcmp(x_category, 'iter')
                if isfield(info_list{alg_idx}, 'iter')
                    x_plot_data = info_list{alg_idx}.iter; 
                elseif isfield(info_list{alg_idx}, 'total_iter')
                    x_plot_data = info_list{alg_idx}.total_iter; 
                end                   
            elseif strcmp(x_category, 'epoch')
                x_plot_data = info_list{alg_idx}.epoch;      
            elseif strcmp(x_category, 'inner_iter')
                x_plot_data = info_list{alg_idx}.subinfos.inner_iter;                  
            elseif strcmp(x_category, 'grad_calc_count')
                x_plot_data = info_list{alg_idx}.grad_calc_count; 
            elseif strcmp(x_category, 'time')
                x_plot_data = info_list{alg_idx}.time;  
            elseif strcmp(x_category, 'lambda') || strcmp(x_category, 'l1-norm')
                x_plot_data = w_list;     
            elseif strcmp(x_category, 'coeff_pos')
                x_plot_data = [1:length(w_list{alg_idx})];                  
            else
            end
            
            
            if strcmp(y_category, 'cost')
                y_plot_data = info_list{alg_idx}.cost;
            elseif strcmp(y_category, 'best_cost')
                y_plot_data = info_list{alg_idx}.best_cost; 
            elseif strcmp(y_category, 'cost_lag')
                y_plot_data = info_list{alg_idx}.cost_lag;                   
            elseif strcmp(y_category, 'optimality_gap')
                y_plot_data = info_list{alg_idx}.optgap;
            elseif strcmp(y_category, 'abs_optimality_gap')
                y_plot_data = info_list{alg_idx}.absoptgap;                
            elseif strcmp(y_category, 'best_optimality_gap')
                y_plot_data = info_list{alg_idx}.best_optgap;                
            elseif strcmp(y_category, 'sol_optimality_gap')
                y_plot_data = info_list{alg_idx}.sol_optgap; 
            elseif strcmp(y_category, 'dual_gap')
                y_plot_data = info_list{alg_idx}.dual_gap;   
            elseif strcmp(y_category, 'const_norm')
                y_plot_data = info_list{alg_idx}.const_norm;  
            elseif strcmp(y_category, 'econst_norm')
                y_plot_data = info_list{alg_idx}.econst_norm;                  
            elseif strcmp(y_category, 'ineconst_norm')
                y_plot_data = info_list{alg_idx}.ineconst_norm;  
            elseif strcmp(y_category, 'inv_rho')
                y_plot_data = info_list{alg_idx}.inv_rho;  
            elseif strcmp(y_category, 'eta')
                y_plot_data = info_list{alg_idx}.eta;                  
            elseif strcmp(y_category, 'gradL_norm')
                y_plot_data = info_list{alg_idx}.gradL_norm;                  
            elseif strcmp(y_category, 'gnorm')
                y_plot_data = info_list{alg_idx}.gnorm;    
            elseif strcmp(y_category, 'subgnorm')
                y_plot_data = info_list{alg_idx}.subgnorm;                  
            elseif strcmp(y_category, 'K')
                y_plot_data = info_list{alg_idx}.K;  
            elseif strcmp(y_category, 'reg') || strcmp(y_category, 'l1-norm') || strcmp(y_category, 'trace_norm')
                y_plot_data = info_list{alg_idx}.reg;  
            elseif strcmp(y_category, 'coeffs') || strcmp(y_category, 'aprox_err')
                y_plot_data = info_list{1};     
            elseif strcmp(y_category, 'coeff_amp') || strcmp(y_category, 'aprox_err')
                y_plot_data = info_list{alg_idx};                     
            elseif strcmp(y_category, 'dnorm')
                y_plot_data = info_list{alg_idx}.subinfos.dnorm;                 
            end
            
            if strcmp(scale_type, 'semilogy')
                semilogy(x_plot_data, y_plot_data, linetype{alg_num}, 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
            elseif strcmp(scale_type, 'loglog')
                loglog(x_plot_data, y_plot_data, linetype{alg_num}, 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;                
            elseif strcmp(scale_type, 'linear')
                if strcmp(y_category, 'coeffs')
                    plot(x_plot_data, y_plot_data, 'Linewidth', linewidth); hold on;
                else
                    plot(x_plot_data, y_plot_data, linetype{alg_num}, 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;  
                end
            else
                error('Invalid scale type');
            end
            
            if ~strcmp(y_category, 'coeff')
                legend_str{alg_num} = algorithm_list{alg_idx};
            end
        else
            %
        end
    end
    hold off;

    % X label
    if strcmp(x_category, 'numofgrad')    
        xlabel('Number of gradient evaluations', 'FontSize', fontsize);
    elseif strcmp(x_category, 'iter')
        xlabel('Iteration', 'FontSize', fontsize);  
    elseif strcmp(x_category, 'epoch')
        xlabel('Epoch', 'FontSize', fontsize);    
    elseif strcmp(x_category, 'grad_calc_count')
        xlabel('Number of gradient calculations', 'FontSize', fontsize);            
    elseif strcmp(x_category, 'time')
        xlabel('Time', 'FontSize', fontsize);   
    elseif strcmp(x_category, 'lambda')
        xlabel('$$\lambda$$', 'FontSize', fontsize,'Interpreter', 'Latex'); 
    elseif strcmp(x_category, 'l1-norm')
        xlabel('$$\ell$$-1 norm', 'FontSize', fontsize,'Interpreter', 'Latex');          
    elseif strcmp(x_category, 'coeff_pos')
        xlabel('Coefficient position', 'FontSize', fontsize); 
    elseif strcmp(x_category, 'inner_iter')
        xlabel('Inner Iteration', 'FontSize', fontsize);         
    else
    end  
    
   
    % Y label    
    if strcmp(y_category, 'cost')    
        ylabel('Cost', 'FontSize', fontsize);
    elseif strcmp(y_category, 'best_cost')    
        ylabel('Best cost', 'FontSize', fontsize);    
    elseif strcmp(y_category, 'cost_lag')    
        ylabel('Lagrangian cost', 'FontSize', fontsize);         
    elseif strcmp(y_category, 'optimality_gap')
        ylabel('Optimality gap', 'FontSize', fontsize);
    elseif strcmp(y_category, 'abs_optimality_gap')
        ylabel('Absolute optimality gap', 'FontSize', fontsize);        
    elseif strcmp(y_category, 'best_optimality_gap')
        ylabel('Best optimality gap', 'FontSize', fontsize);        
    elseif strcmp(y_category, 'sol_optimality_gap')
        ylabel('Solution optimality gap', 'FontSize', fontsize);  
    elseif strcmp(y_category, 'dual_gap')
        ylabel('Dual gap', 'FontSize', fontsize); 
    elseif strcmp(y_category, 'const_norm')
        ylabel('Norm of constraints', 'FontSize', fontsize); 
    elseif strcmp(y_category, 'econst_norm')
        ylabel('Norm of equality constraints', 'FontSize', fontsize);          
    elseif strcmp(y_category, 'ineconst_norm')
        ylabel('Norm of inequality constraints', 'FontSize', fontsize);          
    elseif strcmp(y_category, 'gradL_norm')
        ylabel('Norm of gradient of Lagrangian', 'FontSize', fontsize);    
    elseif strcmp(y_category, 'inv_rho')
        ylabel('Inverse of \rho', 'FontSize', fontsize);  
    elseif strcmp(y_category, 'eta')
        ylabel('\eta', 'FontSize', fontsize);           
    elseif strcmp(y_category, 'gnorm')
        ylabel('Norm of gradient', 'FontSize', fontsize);   
    elseif strcmp(y_category, 'subgnorm')
        ylabel('Norm of subgradient', 'FontSize', fontsize);          
    elseif strcmp(y_category, 'K')
        ylabel('Batch size', 'FontSize', fontsize);   
    elseif strcmp(y_category, 'reg')
        ylabel('Regularizer', 'FontSize', fontsize);  
    elseif strcmp(y_category, 'trace_norm')
        ylabel('Trace (nuclear) norm', 'FontSize', fontsize);            
    elseif strcmp(y_category, 'l1-norm')
        ylabel('$$\ell$$-1 norm', 'FontSize', fontsize, 'Interpreter', 'Latex', 'FontName','Arial'); 
    elseif strcmp(y_category, 'coeffs')
        ylabel('Coefficient', 'FontSize', fontsize);    
    elseif strcmp(y_category, 'aprox_err')
        ylabel('Approximation error', 'FontSize', fontsize); 
    elseif strcmp(y_category, 'coeff_amp') 
        ylabel('Coefficient amplitude', 'FontSize', fontsize);  
    elseif strcmp(y_category, 'dnorm')
        ylabel('Norm of direction', 'FontSize', fontsize);         
    end
    
    
    % range
    if ~isempty(xlim_range)
        xlim([xlim_range])
    end  
    
    if ~isempty(ylim_range)
        ylim([ylim_range])
    end
    
    % legend
    if ~strcmp(y_category, 'coeffs')
        legend(legend_str);
    end
    
    set(gca, 'FontSize', fontsize);      
end

