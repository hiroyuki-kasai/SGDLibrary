function [] = test_polyhedron_quadratic()
    clc; 
    clear; 
    close all;


    %% set problem
    n = 2;
    Q = [0.3, -0.7;-0.7, 2.0];
    p  = [-1; .1];


    G = [0.7, 0.3; 0.3, 1; -1.2, 0; -0.5, -0.7; 0.8, -0.5];
    h = [0.7, 1.2, 0.8, 0.8, 1.0]';
    %A = [ 0.5, 0.3; 0.5, 1; -1, 0; -0.6, -0.7; 0.8, -0.5];
    %b = [0.8, 1.9, 1.0, 1.0, 1.0]';

    problem = constrained_quadratic(Q, -p, [], [], G, h);


    % Plot contour and the constrained region (polyhedron)
    figure
    clf
    hold on
    box on
 
    % draw contour
    x_range_max = 2.5;
    x_range_min = -1.5;
    y_range_max = 2;
    y_range_min = -2;
    unit_len = 1/50;
    xCoarse = x_range_min:unit_len:x_range_max;
    yCoarse = y_range_min:unit_len:y_range_max;
    [XX,YY] = meshgrid(xCoarse, yCoarse); 
    row_size = size(XX,1);      
    col_size = size(XX,2);
    for j=1:col_size
        for i=1:row_size
           w = [XX(i,j); YY(i,j)];
           ZZ(i,j) = problem.cost(w);
        end
    end    
    contour(XX,YY,ZZ,50) 

    % draw constrained region (polyhedron)
    plotregion(-G,-h,[],[],[0.8,0.8,0.8]); % plot the feasible region

    % set region and labels
    xlim([x_range_min x_range_max])
    ylim([y_range_min y_range_max])
    xlabel('x_1')
    ylabel('y_2')
  


    %% calcualte solution by quadprog
    [w_opt, ~] = quadprog(Q, p, G, h);
    f_opt = problem.cost(w_opt);


    %% Performe projected gradinet descent (PGD)
    % general options for optimization algorithms  
    clear options;
    options.w_init = [-1; 1];
    options.tol_gnorm = 1e-10;
    options.max_epoch = 15;
    options.verbose = true;  
    options.f_opt = f_opt;        
    options.store_w = true;
    options.store_grad = true;
    % options.step_alg = 'backtracking';
    % options.backtracking_c = 1/2;
    % options.backtracking_rho = 1/2;
    options.step_alg = 'decay-1';
    options.step_init = 1;
    % perform PGD
    [pgd, info_list] = sd(problem, options);

    fprintf('Solution: QuadProg: (%.2f, %.2f), PGD:  (%.2f, %.2f)\n', w_opt(1), w_opt(2), pgd(1), pgd(2));


    %% Plot
    for i = 1:size(info_list.w, 2)
        plot(info_list.w(1,i),info_list.w(2,i),'-ko','MarkerSize',8)
        quiver(info_list.w(1,i),info_list.w(2,i),-info_list.grad(1,i),-info_list.grad(2,i),0, 'LineWidth', 1)
        pause(0.2)
    end

    % plot solution
    plot(w_opt(1,:),w_opt(2,:),'pentagram', 'MarkerSize',20, 'Markerfacecolor', 'green')
end

