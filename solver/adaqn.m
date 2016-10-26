function [w, infos] = adaqn(problem, options)
% Adaptive Quasi-Newton Algorithm (AdaQN) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       Nitish Shirish Keskar and Albert S. Berahas,
%       "adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs," 
%       European Conference, ECML PKDD 2016, 2016.
%
%                   
% Created by H.Kasai on Oct. 17, 2016


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % extract options
    if ~isfield(options, 'step')
        step_init = 0.1;
    else
        step_init = options.step;
    end
    step = step_init;
    
    if ~isfield(options, 'step_alg')
        step_alg = 'fix';
    else
        if strcmp(options.step_alg, 'decay')
            step_alg = 'decay';
        elseif strcmp(options.step_alg, 'fix')
            step_alg = 'fix';
        else
            step_alg = 'decay';
        end
    end     
    
    if ~isfield(options, 'lambda')
        lambda = 0.1;
    else
        lambda = options.lambda;
    end 
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end        

    if ~isfield(options, 'batch_size')
        batch_size = 1;
    else
        batch_size = options.batch_size;
    end
    num_of_bachces = floor(n / batch_size);        
    
    if ~isfield(options, 'max_epoch')
        max_epoch = inf;
    else
        max_epoch = options.max_epoch;
    end 

    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end     
    
    if ~isfield(options, 'L')
        L = 20;
    else
        L = options.L;
    end     
        
    if ~isfield(options, 'r')
        r = 20;
    else
        r = options.r;
    end 
    
    if ~isfield(options, 'fisher_mem_length')
        fisher_mem_length = 20;
    else
        fisher_mem_length = options.fisher_mem_length;
    end      
    
    if ~isfield(options, 'f_sol')
        f_sol = -Inf;
    else
        f_sol = options.f_sol;
    end     
    
    if ~isfield(options, 'permute_on')
        permute_on = 1;
    else
        permute_on = options.permute_on;
    end     
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end    
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    s_array = [];
    y_array = [];    
    w_s = zeros(d,1);
    w_o = zeros(d,1);  
    % create the Fisher container
    fisher_container = [];    

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_sol;
    infos.optgap = optgap;
    infos.cost = f_val;
    
    % randomly sample the data points labeled 'monitoring set'
    indices_monitor = randperm(n,options.batch_size);    
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % permute samples
        if permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end
        
        for j=1:num_of_bachces
            
            % update step-size
            if strcmp(step_alg, 'decay')
                step = step_init / (1 + step_init * lambda * total_iter);
            end                  
         
            % calculate averaged w
            w_s = w_s + w/L;
            
            % calculate gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            
            % append the new stochastic gradient to the Fisher
            % container
            fisher_container = [fisher_container grad];
            % if size of the container exceeds the capacity, delete the
            % oldest stochastic gradient stored
            if(size(fisher_container,2)>fisher_mem_length)
                fisher_container(:,1) = [];
            end
            % add the current iterate to the running sum over an epoch            
            
            % perform LBFGS two loop recursion
            Hg = lbfgs_two_loop_recursion(grad, s_array, y_array);
            % update w            
            w = w + (step*Hg);            

            % update 
            if(mod(total_iter,L)==0)   
                
                w_n = w_s;
                w_s = zeros(d,1);                
                
                if total_iter  

                    % if new function value is 1.01 times lareger than
                    % the old function value, reset curvature pairs
                    f_val = problem.cost_batch(w_n,indices_monitor);
                    fval_old = problem.cost_batch(w_o,indices_monitor);
                    if(f_val > 1.01*fval_old)
                        % reset S and Y
                        s_array = [];
                        y_array = []; 
                        w = w_o;
                        continue;
                    end
                    % update the curvature pairs
                    s = w_n - w_o;
                    y = fisher_container * (fisher_container' * s);
                    % if curvature estimate is less than sqrt of
                    % machine precision, SKIP the curvature pair update
                    rho = dot(s,y)/dot(y,y);
                    if(rho>1e-4)
                        % store cavature pair
                        % 'y' curvature pair is calculated from a Hessian-vector product.
                        s_array = [s_array s];
                        y_array = [y_array y]; 

                        % remove overflowed pair
                        if(size(s_array,2)>r)
                            s_array(:,1) = [];
                            y_array(:,1) = [];
                        end  
                        
                        w_o = w_n;
                    else
                        %fprintf('SKIP \n');
                    end               
                else
                    % update the old averaged uterate
                    w_o = w_n;
                end  
               
            end
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * batch_size;        
        % update epoch
        epoch = epoch + 1;
        % calculate optgap
        f_val = problem.cost(w);
        optgap = f_val - f_sol;        

        % store infos
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];

        % display infos
        if verbose > 0
            fprintf('adaQN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end      
end

