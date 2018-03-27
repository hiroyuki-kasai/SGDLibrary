function [w, infos] = adagrad(problem, in_options)
% AdaGrad and RMSProp algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       sub_mode:   AdaGrad
%                   John Duchi, Elad Hazan and Yoram Singer, 
%                   ?gAdaptive Subgradient Methods for Online Learning and Stochastic Optimization,?h 
%                   The Journal of Machine Learning Research 12: 2121-2159, 2011.
%       sub_mode:   RMSProp
%                   Tijmen Tieleman and G. Hinton,
%                   Lecture 6.5 - rmsprop, 
%                   COURSERA: Neural Networks for Machine Learning, 2012.
%       sub_mode:   AdaDelta
%                   M. D.Zeiler, 
%                   "AdaDelta: An Adaptive Learning Rate Method," 
%                   arXiv preprint arXiv:1212.5701, 2012.
%                   
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 17, 2016
% Modified by H.Kasai on Mar. 25, 2018
% Note that partial code is originaly created by M.Pak (See https://github.com/mp4096/adawhatever)


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options 
    local_options.sub_mode = 'AdaGrad';
    local_options.beta = 0.9;
    local_options.epsilon = 1e-8;
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);     
    
    % initialise moment estimates
    if strcmp(options.sub_mode, 'AdaGrad')    
        gradHist = zeros(d, 1);  
    elseif strcmp(options.sub_mode, 'RMSProp')    
        r = zeros(d, 1);
    else % 'AdaDelta'
        r = zeros(d, 1);
        s = zeros(d, 1);        
    end   

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);     
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
    end    
    
    % set start time
    start_time = tic();    

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        % permute samples
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end
        
        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);            
         
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w,indice_j);
            
            if strcmp(options.sub_mode, 'AdaGrad')
                % update historical gradients
                gradHist = gradHist + grad.^2;
                
                % update w
                w = w - step * grad ./ (sqrt(gradHist) + options.epsilon);                  
            elseif strcmp(options.sub_mode, 'RMSProp')
                if ~total_iter
                    r = grad.^2;  
                else
                    % calculate accumulate squared gradient
                    r = options.beta * r + (1-options.beta)* grad.^2;                    
                end
                
               % update w
                w = w - step * grad ./ (sqrt(r) + options.epsilon);                   
            else % 'AdaDelta'
                if ~total_iter
                    r = grad.^2;
                else
                    % calculate accumulate squared gradient
                    r = options.beta * r + (1-options.beta)* grad.^2;                    
                end
                
                % update
                v = - (sqrt(s) + options.epsilon)./(sqrt(r) + options.epsilon) .* grad;
                % update accumulated updates (deltas)
                s = options.beta * s + (1-options.beta)* v.^2;

               % update w
                w = w + v;                   
            end
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
                
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * options.batch_size;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);         

        % display infos
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end    
end

