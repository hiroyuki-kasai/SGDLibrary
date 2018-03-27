function  convergence_animation_demo()
% demonstration file for GDLibrary.
%
% This file illustrates how to use this library in case of "Rosenbrock" 
% problem. This demonstrates GD, NCG and L-BFGS algorithms.
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Nov. 02, 2016


    clc;
    clear;
    close all;


    %% define problem definitions
    % set number of dimensions
    d = 2;    
    problem = rosenbrock(d);
    
    
    %% calculate solution 
    w_opt = problem.calc_solution(); 

   
    %% general options for optimization algorithms   
    options.w_init = zeros(d,1);
    % set gradient norm tolerance    
    options.tol_gnorm = 1e-10;
    % set max epoch
    options.max_iter = 100;
    % set verbose mode        
    options.verbose = true;  
    % set solution    
    options.f_opt = problem.cost(w_opt);  
    % set store history of solutions
    options.store_w = true;
  
    
    %% perform GD with backtracking line search 
    options.step_alg = 'backtracking';
    [~, info_list_gd] = sd(problem, options); 
    
    %% perform GD with backtracking line search 
    options.step_alg = 'backtracking';
    [~, info_list_ncd] = ncg(problem, options);     

    %% perform L-BFGS with strong wolfe line search
    options.step_alg = 'strong_wolfe';                  
    [~, info_list_lbfgs] = lbfgs(problem, options);                  
    
    
    %% plot all
    close all;
    
    % draw convergence animation   
    w_history = cell(1);
    w_history{1} = info_list_gd.w;
    w_history{2} = info_list_ncd.w;  
    w_history{3} = info_list_lbfgs.w;      
    speed = 0.5;
    draw_convergence_animation(problem, {'SD-BKT', 'NCG-BKT', 'LBFGS-WOLFE'}, w_history, options.max_iter, speed);    
    
end


