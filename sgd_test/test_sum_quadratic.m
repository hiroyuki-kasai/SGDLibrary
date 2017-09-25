function  test_sum_quadratic()

    clc;
    clear;
    close all;

    
    %% Set algorithms
    if 0
        algorithms = sgd_solver_list('ALL');  
    else
        algorithms = {'SGD','SVRG','IQN'};
    end
    
    
    % prepare datasets
    N = 1000;   % number of random functions 
    d = 100;    % dimension 

    A = zeros(d,d,N);
    b = zeros(d,N);

    for i=1:N
        for j=1:(d/2)
           %A(j,j,i)=10^(0+floor(3*rand(1)));  % good Condition number 
           A(j,j,i)=10^(-2+floor(3*rand(1))); %  bad Condition number 
        end
        for j=((d/2)+1):d
            A(j,j,i)=10^(2+floor(3*rand(1)));
        end

    end

    for i=1:N
        for j=1:d
            b(j,i)=1000*rand(1);
        end
    end

    A_sum=zeros(d,d);
    for i=1:N
        A_sum=A_sum+A(:,:,i);
    end

    b_sum=zeros(d,1);
    for i=1:N
        b_sum=b_sum+b(:,i);
    end

    cn = max(eig(A_sum))/min(eig(A_sum));
    fprintf('condition number: %e\n', cn);


    % define problem definitions
    problem = sum_quadratic(A, b);

    % Calculate the solution
    A_inv=zeros(d,d);
    for i=1:d
        A_inv(i,i)=1/(A_sum(i,i));
    end

    w_opt=-A_inv*b_sum;
    f_opt = problem.cost(w_opt); 
    fprintf('%f\n', f_opt); 

    
    %% initialize
    w_init = zeros(d,1);
    batch_size = 1;
    w_list = cell(length(algorithms),1);
    info_list = cell(length(algorithms),1);       

    
    %% perform algorithms
    for alg_idx=1:length(algorithms)
        fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
        
        clear options;
        % general options for optimization algorithms   
        options.w_init = w_init;
        options.tol_optgap = 1e-12;
        options.max_iter = 100;
        options.verbose = true;   
        options.f_opt = f_opt;        
        options.store_w = false;
        options.permute_on = 1;    
        options.f_opt = f_opt; 
        options.batch_size = batch_size;
        

        switch algorithms{alg_idx}
            
           case {'SGD'} 

                options.step_init = 0.0001 * options.batch_size;
                %options.step_alg = 'decay';
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);   
                
           case {'IQN'} 

                options.step_init = 1;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = iqn(problem, options);   
                
            case {'SVRG'}
                
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';

                [w_list{alg_idx}, info_list{alg_idx}] = svrg(problem, options);      
                
            case {'SAG'}
                
                options.step_init = 0.000001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAG';               

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);      
                
            case {'SAGA'}
                
                options.step_init = 0.00000001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SAGA';                       

                [w_list{alg_idx}, info_list{alg_idx}] = sag(problem, options);   
                
            case {'SVRG-SQN'}       
 
                options.batch_hess_size = batch_size * 20;        
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-SQN';
                options.L = 20;
                options.r = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);
                
            case {'SVRG-LBFGS'}                  
 
                options.step_init = 0.0001 * options.batch_size;
                options.step_alg = 'fix';
                options.sub_mode = 'SVRG-LBFGS';
                options.mem_size = 20;

                [w_list{alg_idx}, info_list{alg_idx}] = slbfgs(problem, options);                   
            
            case {'Newton-CHOLESKY'}

                options.sub_mode = 'CHOLESKY';                
                options.step_alg = 'backtracking';
                [w_list{alg_idx}, info_list{alg_idx}] = newton(problem, options);

            otherwise
                warn_str = [algorithms{alg_idx}, ' is not supported.'];
                warning(warn_str);
                w_list{alg_idx} = '';
                info_list{alg_idx} = '';                
        end
        
    end
    
    
    %% plot all
    close all;
    
    % display iter vs cost/gnorm
    display_graph('iter','optimality_gap', algorithms, w_list, info_list);
    display_graph('time','optimality_gap', algorithms, w_list, info_list);    
    display_graph('iter','gnorm', algorithms, w_list, info_list);  
     

end


