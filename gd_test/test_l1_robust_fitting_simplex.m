function [] = test_l1_robust_fitting_simplex()

    % min f(w) = || A * w - b ||_1,   s.t. w in probablity simplex.

    close all
    clear 
    clc

    %randn('seed',315);

    %% generate dataset
    n = 50; 
    d = 50; 
    A = randn(d,n); 
    beta = randn(n,1); 
    beta_opt = proj_simplex(beta, 1, 'eq');
    noise = 0.0;
    b = A * beta_opt + noise * randn(d, 1) ;


    %% define problem definitions
    problem = l1_robust_fitting_simplex(A, b); 
    f_opt = problem.cost(beta_opt);
    
    %% set initial solution
    w_init = ones(n, 1)/n;
    maxIter = 500;
 

    %% Perform Subgradient method
    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.max_epoch = maxIter;
    options.f_opt = f_opt;
    options.verbose = true; 
%     options.step_alg = 'fix';
%     options.step_init = 0.01;
%     R = sqrt(2);
%     L = sqrt(d)*norm(A);
%     eta0 = R/(L*sqrt(maxIter)); 
%     options.step_init = eta0;  
    options.linesearchfun = @subgd_stepalg;  % set my_stepalg (user-defined stepsize algorithm)     
    algorithms{1} = 'Projected Subgradient';
    [w_list{1}, info_list{1}] = subgrad(problem, options);


    %% Perform Mirror descent method    


    % re-define problem definitions
    clear problem
    problem = l1_robust_fitting_simplex(A, b, 'bregman'); 

    clear options;
    options.w_init = w_init;
    options.tol_gnorm = -inf;
    options.max_epoch = maxIter;
    options.f_opt = f_opt;
    options.verbose = true; 
%     options.step_alg = 'fix';
%     options.step_init = 0.01;
%     R = sqrt(log(n));
%     maX = -1;
%     for i = 1:n
%         maX = max(maX, norm(A(:,i),1));
%     end
%     kappa = 1.0;
%     eta0 = sqrt(2*kappa/maxIter)*R/maX;
%     options.step_init = eta0;
    options.linesearchfun = @md_stepalg;  % set my_stepalg (user-defined stepsize algorithm)    
    algorithms{2} = 'Mirror Descent';     
    [w_list{2}, info_list{2}] = md(problem, options);
    

    %% plot all
    % display iter vs. cost
    display_graph('iter', 'cost', algorithms, w_list, info_list);
    % display iter vs. best_cost
    display_graph('iter', 'best_cost', algorithms, w_list, info_list);    
    % display iter vs. optimality_gap
    %display_graph('iter', 'optimality_gap', algorithms, w_list, info_list);
    % display iter vs. best_optimality_gap
    %display_graph('iter', 'best_optimality_gap', algorithms, w_list, info_list);    
    
end


%% define user-defined stepsize algorithm
function [step, nothing] = subgd_stepalg(step_alg, problem, w, w_old, grad, grad_old, prev_step, options)
    iter = options.iter;
    subgrad = options.subgrad;
    step = sqrt(2)/( norm(subgrad) * sqrt(iter+1));
    nothing = [];
end  

function [step, nothing] = md_stepalg(step_alg, problem, w, w_old, grad, grad_old, prev_step, options)
    iter = options.iter;
    subgrad = options.subgrad;
    step = sqrt(2)/( norm(subgrad,'Inf') * sqrt(iter+1));
    nothing = [];
end   




