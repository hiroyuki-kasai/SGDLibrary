function [w, infos]= tr(problem, in_options)
% Trust region medhod algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
%
%       H_mode:   
%                   "NEWTON"
%                           Hessian HESS of f is used
%                   "QUASI-NEWTON"           
%                           Rank-one updates approximations of the Hessian are
%                           built as in BFGS and the variable HESS is not required.
%
%       subprob_solver:   
%                   "Trustone"
%                   "Cauchy_point"           
%                   "Dogleg"
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% This file was originally created in the book:
%   A. Quarteroni, F. Saleri, P. Gervasio
%   "Scientific Computing with MATLAB and Octave"
%   Springer Science & Business Media, 2014/02/20
%
% Sub-program solvers, dogleg and cauchy point, are originally from the codes 
%   written by Philip D. Loewen.
%   http://www.math.ubc.ca/~loew/m604/mfiles.htm
%
%
%
% TAdjusted by H.Kasai on Apr. 08, 2018
% Modified by H.Kasai on Mar. 23, 2018


    % set dimensions and samples
    d = problem.dim;
    n = problem.samples;     
    
    % set local options 
    local_options = []; 
    local_options.algorithm = 'TR'; 
    local_options.subprob_solver = 'DOGLEG';
    local_options.H_mode = 'QUASI-NEWTON';
    local_options.delta = 0.5;
    local_options.mu = 0.1;    
    local_options.eta1= 0.25;  
    local_options.eta2 = 0.75; 
    local_options.gamma1 = 0.25; 
    local_options.gamma2 = 2;    
    local_options.delta_max = 5;     

    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);  


    % initialise
    iter = 0;
    grad_calc_count = 0;
    w = options.w_init;    
    
   % initialize by BB step-size 
    if strcmp(options.step_init_alg, 'bb_init')
        options.step_init = bb_init(problem, w);
    end    
    
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    grad_old = [];   
    
    
    %gk = grad_func(w); 
    gk = problem.grad(w);
    eps2 = sqrt(eps);

    if strcmp(options.H_mode, 'NEWTON')
        %Hk=hess(w);
        Hk = problem.hess(w);
    else
        Hk = eye(length(w)); 
    end    
    
    % display infos
    if options.verbose
        fprintf('TR (%s,%s) %s %s : Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', options.subprob_solver, options.H_mode, '   ', '   ', iter, f_val, gnorm, optgap);
    end      
    
    
    % set start time
    start_time = tic();  
    

    while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)         
        
        % sub-problem calculates the proposed step "s".
        if strcmp(options.subprob_solver, 'TRUSTONE')
            s = trustone(Hk, gk, options.delta);
        elseif strcmp(options.subprob_solver, 'CAUCHY')
            s = cauchy_point(Hk, gk, options.delta);
        elseif strcmp(options.subprob_solver, 'DOGLEG')
            s = dogleg(Hk, gk, options.delta);
        elseif strcmp(options.subprob_solver, 'CG')
            params(1) = 1e-8;
            params(2) = 10;
            params(3) = 1;
            [s, ~, ~] = cg_steihaug(Hk, gk, options.delta, params);
%         elseif strcmp(subprob_solver, 'Lanczos')
%             s = lstrs(Hk, gk, delta);
        end

        rho = (problem.cost(w+s)-problem.cost(w))/(s'*gk+0.5*s'*Hk*s);
        
        % judge whether accept or reject the proposed step "s" based on the model performance.
        if rho > options.mu 
            w1 = w + s; 
            accstr = 'ACC';
        else 
            w1 = w; 
            accstr = 'REJ';
        end

        % update the trust region radius: delta
        if rho < options.eta1
            % If the decrease is less than 1/4 of the predicted decrease,
            % we then reduce the trust region (TR) radius.            
            options.delta = options.gamma1 * options.delta;
            trstr = 'TR-';
        elseif rho > options.eta2 && abs(norm(s)-options.delta)<sqrt(eps)
            % If the decrease is greter than 3/4 of the precicted decrease and
            % "s" is close to the trust region boundary, we increase the TR radius.
            options.delta = min([options.gamma2*options.delta,options.delta_max]); 
            trstr = 'TR+';
        else
            % Otherwise, we keep the trust region radius unchanged. 
            trstr = '   ';
        end

        gk1 = problem.grad(w1); 
        %err = norm((gk1.*w1)/max([abs(fun(w1)),1]),inf);
        %err = norm((gk1.*w1)/max([abs(problem.cost(w1)),1]),inf);

        if strcmp(options.H_mode, 'NEWTON')
            % Exact Newton
            w = w1;
            gk = gk1; 
            Hk = problem.hess(w);
        else          
            % quasiNewton
            gk1 = problem.grad(w1);
            yk = gk1-gk; 
            sk = w1-w; 
            yks = yk'*sk;
            if yks > eps2*norm(sk)*norm(yk) 
                Hs = Hk*sk; 
                Hk = Hk+(yk*yk')/yks-(Hs*Hs')/(sk'*Hs);
            end
            w = w1; 
            gk = gk1;
        end

        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        
        % store infos
        [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);        

        % display infos
        if options.verbose
            fprintf('TR (%s,%s) %s %s : Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e\n', options.subprob_solver, options.H_mode, accstr, trstr, iter, f_val, gnorm, optgap);
        end        
    end

    if gnorm < options.tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', options.tol_gnorm);
    elseif optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);        
    elseif iter == options.max_epoch
        fprintf('Max iter reached: max_epoch = %g\n', options.max_epoch);
    end  
end

function s = trustone(Hk, gk, delta) 

    if 0
        s = -Hk\gk; 
    else
        L = chol(Hk,'lower');
        y = L\gk;
        s = -L'\y;    
    end
    
    d = eigs(Hk,1,'sa');
    
    if norm(s)>delta || d<0 
        lambda = abs(2*d); 
        I = eye(size(Hk)); 
        for l=1:3
            R = chol(Hk+lambda * I);
            s = -R \ (R'\gk); 
            q = R'\s; 
            lambda = lambda + (s'*s)/(q'*q)*(norm(s)-delta)/delta; 
            if lambda < -d 
                lambda = abs(lambda*2); 
            end
        end
    end
end


function s = dogleg(Hk, gk, delta) 

    if 0
        pB = -Hk\gk; 
    else
        L = chol(Hk,'lower');
        y = L\gk;
        pB = -L'\y;    
    end
    
    predredB = -(gk'*pB + 0.5*pB'*Hk*pB); % Model reduction for Newton Pt

    gHg = gk' * Hk * gk;  
  
    if predredB > 0
        % Good situation: Newton point (pB) reduces the model function.
        pBlen = norm(pB);
        
        if pBlen <= delta, % Newton point in trust region: use it!
            s = pB;
        else    
            % find unconstrained steepest descent (sd) minimizer (pC) for Model Function.
            pC = -(gk'*gk)/gHg*gk;
            
            if norm(pC) >= delta,
                % if it is outside the TR, return the point where the path intersects the boundary.
                % Both Newton and Cauchy (SD) points lie outside, use Cauchy point on boundary.
                s = - delta * gk/norm(gk);
            else
                % SD point (pC) is inside while Newton point (pB) is outside of trust region.
                % Find boundary point of trust region on line joining them.
                pB_pSD = pB - pC;
                aa = pB_pSD'*pB_pSD;
                bb = 2*pB_pSD'*pC;
                cc = pC'*pC - delta^2;
                t  = (-bb+(bb^2-4*aa*cc)^0.5)/2/aa;
                s = pC + t*pB_pSD;
            end
        end
    
    else  % Bad situation: Newton popint (pB) *increases* model function
  
        if gHg > 5*eps,
            % 1D function is convex enough:  
            % Compute unconstrained Cauchy Point, i.e., unconstrained Steepest Descent Minimizer for Model Function.
            pC = -(gk'*gk)/gHg*gk;

            if norm(pC) <= delta,
                s = pC;
            else
                s = delta*pC/pC_norm;
            end;
        else
            % 1D slice of model function in gradient direction is concave:
            % SD minimizer lies on TR boundary, in direction of -g'
            s = - delta * gk/norm(gk);
        end
        
    end
end


function s = cauchy_point(Hk, gk, delta) 

    gHg = gk' * Hk * gk;
  
    if gHg > 5*eps,
        % 1D function is convex enough:  
        % Find Unconstrained Steepest Descent Minimizer for Model Function.
        pC = -(gk'*gk)/gHg*gk;
        pC_norm = norm(pC);
        
        if pC_norm <= delta,
            s = pC;
        else
            s = delta*pC/pC_norm;
        end;
    else
        % 1D slice of model function in gradient direction is concave:
        % SD minimizer lies on TR boundary, in direction of -g'
        s = - delta*gk/norm(gk);
    end
end 


%--------------------------------------------------------------------------
%
function [p, num_cg, iflag] = cg_steihaug(G, b, delta, params)
    % This file is written by Jose Luis Morales.
    % from http://www.ece.northwestern.edu/~morales/.
    %
    %--------------------------------------------------------------------------         
    %
    % ... This procedure approximately solves the following trust region 
    %     problem 
    %   
    %         minimize    Q(p) = 1/2 p'Gp + b'p  
    %         subject to  ||p|| <= Delta                      
    %
    %
    %     by means of the CG-Steihaug method. 
    %
    %--------------------------------------------------------------------------
    % INPUT
    %
    %             G:  Hessian matrix
    %             b:  gradient vector
    %         delta:  radius of the TR
    %     params(1):  relative residual reduction factor
    %     params(2):  max number of iterations
    %     params(3):  level of output
    %
    % OUTPUT
    %             p:  an aproximate solution of (1)-(2)
    %        num_cg:  number of CG iterations to achieve convergence
    %         iflag:  termination condition 
    %
    %--------------------------------------------------------------------------         
    %
    n      = length(b);  
    errtol = params(1); 
    maxit  = params(2); 
    iprnt  = params(3); 
    iflag  = ' ';
    %
    g  = b; 
    x  = zeros(n,1); r = -g;
    %
    z   = r;
    rho = z'*r;   
    tst = norm(r); 
    flag  = ''; 
    terminate = errtol*norm(g);   
    it = 1;    
    hatdel = delta*(1-1.d-6);
    rhoold = 1.0d0;
    if iprnt > 1 
        fprintf(1,'\n\tThis is an output of the CG-Steihaug method. \n\tDelta = %7.1e \n', delta);
        fprintf(1,'   ---------------------------------------------\n');
    end
    flag = 'We do not know ';
    if tst <= terminate; flag  = 'Small ||g||    '; end 

    while((tst > terminate) & (it <=  maxit) & norm(x) <=  hatdel)
        %
        if(it == 1) 
            p = z;
        else
            beta = rho/rhoold;
            p = z + beta*p;
        end
        %
        % 
        w  = G*p;  alpha = w'*p;
        %
        % If alpha < = 0 head to the TR boundary and return
        %
        ineg = 0;
        if(alpha <=  0)
            ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
            alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
            flag  = 'negative curvature';
            iflag = 'NC';
        else
            alpha = rho/alpha;
            if norm(x+alpha*p) > delta
                ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
                alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
                flag  = 'boundary was hit';
                iflag = 'TR';
            end
        end
        x   =  x + alpha*p;
        r   =  r - alpha*w;
        tst = norm(r);
        if tst <= terminate; flag = '||r|| < test   '; iflag = 'RS'; end;
        if norm(x) >=  hatdel; flag = 'close to the boundary'; iflag = 'TR'; end

        if iprnt > 0 
            fprintf(1,' %3i    %14.8e   %s  \n', it, tst, flag);
        end
        rhoold = rho;
        z   = r; 
        rho = z'*r;
        it  = it + 1;
    end %           
    if it > maxit; iflag = 'MX'; end;

    num_cg = it;
    p = x;
end

