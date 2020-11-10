function [] = test_one_dim_denosing()

    clc;
    clear;
    close all;
    
    %rng('default')
    %rng(10)
    
    max_epoch = 100;
    verbose = 2;
    lambda_array = [0.1, 1, 10, 100];
    
    
     
    %% Set algorithms
    algorithms = {'SD-BKT', 'AG-BKT'};
    
    
    
    %% generate dataset
    % number of samples
    n = 100;
    noise_std = 0.1;

    if 0
        % Original signal
        x = linspace(0,4, n)';
        y = sin(x) + x.*(cos(x).^2);

        % Noisy signal
        y_noise = y + noise_std * randn(n, 1);
    else
        % Original signal
        x = linspace(0, 4, n)';
        y = zeros(n, 1);
        y(1:25) = 1;
        y(26:50) = 3;
        y(51:100) = 2;
        
        % Noisy signal        
        y_noise = y + noise_std * randn(n, 1);        
        
    end

    
    %% initialize
    w_list = cell(1);    
    info_list = cell(1);          
           
    total_idx = 0;
    
    for lamda_idx = 1 : numel(lambda_array)
    
        %% define problem definitions
        problem = one_dim_denoise_problem(y_noise, lambda_array(lamda_idx));
        w_init = randn(problem.d, 1);


        %% calculate optimal solution for optimality gap
        cal_sol_options.verbose = verbose;
        [w_opt] = problem.calc_solution(w_init, cal_sol_options);    


        %% perform algorithms
        for alg_idx=1:length(algorithms)
            
            total_idx = total_idx + 1;
            
            %% perform algorithm
            clear options;
            options.w_init = w_init;
            options.tol_gnorm = 1e-10;
            options.tol_optgap = 1e-20;
            options.max_iter = max_epoch;
            options.w_opt = w_opt;
            options.verbose = true; 
            options.store_w = true;
            options.f_opt = problem.cost(w_opt);    

            % GD
            options.step_alg = 'backtracking';
            [w_list{total_idx}, info_list{total_idx}] = sd(problem, options);  

            % AG
            options.step_alg = 'backtracking';
            [w_list{total_idx+1}, info_list{total_idx+1}] = ag(problem, options);      

        end                
        fprintf('\n\n');
        
    end


  


    
    %% plot all
    close all;
    
%     % display iter vs cost/gnorm
%     display_graph('iter','cost', algorithms, w_list, info_list);
%     % display iter vs cost/gnorm
%     display_graph('iter','optimality_gap', algorithms, w_list, info_list);    
%     % display iter vs. l1 norm, i.e. the toral number of non-zero elements 
%     %display_graph('iter','reg', algorithms, w_list, info_list); 
%     
%     figure
%     plot(x, y, ':k', 'LineWidth', 2); hold on 
%     plot(x, y_noise, 'b', 'LineWidth', 2); hold on 
%     plot(x, w_out('w'), 'r', 'LineWidth', 2); hold off 
%     legend('Original signal', 'Noisy signal', 'Denoised signal');
%     title('Denoised signal')    
%     
%     
%     denoised_signal = cell(numel(algorithms), 1);
%     for k=1:numel(algorithms)
%         denoised_signal{k} = w_list{k};
%     end
    
    
    denoised_signal = cell(numel(total_idx), 1);
    total_idx = 0;
    for lambda_idx = 1 : numel(lambda_array)
        for alg_index = 1 : numel(algorithms)
            total_idx = total_idx + 1;
            denoised_signal{total_idx} = w_list{total_idx};
        end
    end

    figure;
    
    width = numel(algorithms);
    height = numel(lambda_array); 
    
    fontsize = 12;
    total_idx = 0;
    for lambda_idx = 1 : numel(lambda_array)
        for alg_index = 1 : numel(algorithms)
            
            total_idx = total_idx + 1;

            %width = alg_index;

            %frame_num_str = sprintf('%03d', frame_num);

            subplot(height, width, total_idx);
            plot(x, y, ':k', 'LineWidth', 2); hold on 
            plot(x, y_noise, 'b', 'LineWidth', 2); hold on 
            plot(x, denoised_signal{total_idx}, 'r', 'LineWidth', 2); hold off 
            legend('Original signal', 'Noisy signal', 'Denoised signal');
            title_str = sprintf('%s (lambda = %5.1f)', algorithms{alg_index}, lambda_array(lambda_idx));
            title(title_str)               
            set(gca, 'FontSize', fontsize);
        end  
    end
    
    
    
end




