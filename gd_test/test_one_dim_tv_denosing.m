function [] = test_one_dim_tv_denosing()

    clc;
    clear;
    close all;
    
    rng('default')
    
    max_epoch = 50;
    lambda_array = [0.1, 1, 10, 100];

    %% generate dataset
    noise_std = 0.1;

    if 0
        % Original signal   
        n = 100;
        x = zeros(n,1);
        x(1:25) = 1;
        x(26:50) = 3;
        x(51:100) = 2;     
        
    else
        % Original signal           
        n = 1000;
        x = zeros(n,1);
        x(1:250) = 1;
        x(251:500) = 3;
        x(751:1000) = 2;
        
    end
    
    y = x + noise_std * randn(n, 1);
    
    algorithms = {'GD', 'AG', 'DPG', 'FDPG'};

    
    %% initialize
    w_list = cell(1);    
    info_list = cell(1);          
           
    total_idx = 0;
    
    for lamda_idx = 1 : numel(lambda_array)
    
        %% define l2-reg problem definitions
        problem_l2 = one_dim_tv_denoise_problem(y, lambda_array(lamda_idx), 'l2');
        w_init = randn(problem_l2.d, 1);
        
        %% perform algorithms
        clear options;
        options.w_init = w_init;
        options.tol_gnorm = 1e-10;
        options.max_epoch = max_epoch;
        options.verbose = true; 

        % GD
        total_idx = total_idx + 1;            
        options.step_alg = 'backtracking';
        [w_list{total_idx}, info_list{total_idx}] = sd(problem_l2, options);  

        % AG
        total_idx = total_idx + 1;             
        options.step_alg = 'backtracking';
        [w_list{total_idx}, info_list{total_idx}] = ag(problem_l2, options);
        
        

        %% define l1-reg problem definitions
        problem_l1 = one_dim_tv_denoise_problem(y, lambda_array(lamda_idx), 'l1');        
        y_init = zeros(problem_l2.d-1, 1);        

        clear options;
        options.w_init = y_init;
        options.tol_gnorm = 1e-10;
        options.max_epoch = max_epoch;
        options.verbose = true; 
        options.step_init = 1/4; % L = 4;
        options.w_init = y_init;

        % dpg
        total_idx = total_idx + 1;             
        options.sub_mode = 'std';
        [w_list{total_idx}, ~, info_list{total_idx}] = dpg(problem_l1, options);  

        % fdpg
        total_idx = total_idx + 1;             
        options.sub_mode = 'fast';
        [w_list{total_idx}, ~, info_list{total_idx}] = dpg(problem_l1, options);                 
              
        fprintf('\n\n');
        
    end

  
    %% plot all
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
            subplot(height, width, total_idx);
            plot(1:n, x, ':k', 'LineWidth', 2); hold on 
            plot(1:n, y, 'b', 'LineWidth', 2); hold on 
            plot(1:n, denoised_signal{total_idx}, 'r', 'LineWidth', 2); hold off 
            legend('Original signal', 'Noisy signal', 'Denoised signal');
            title_str = sprintf('%s (lambda=%5.1f)', algorithms{alg_index}, lambda_array(lambda_idx));
            title(title_str)               
            set(gca, 'FontSize', fontsize);
        end  
    end
    
end




