function [s, lambda_k] = cr_subsolver(problem, w, grad, sub_hess_indices, sigma, ...
                                              successful_flag, lambda_k, subproblem_solver,...
                                              exact_tol, krylov_tol, solve_each_i_th_krylov_space,...
                                              keep_Q_matrix_in_memory)
% Original Python code was created by J. M. Kohler and A. Lucchi (https://github.com/dalab/subsampled_cubic_regularization)
%
%
% This file is part of SGDLibrary.
%
% Ported to MATLAB code by K.Yoshikawa and H.Kasai on March, 2018.
% Modified by H.Kasai on Apri. 05, 2018                                          
                                          
                                          
    if strcmp(subproblem_solver, 'cauchy_point')
        % min m(-a*grad) leads to finding the root of a quadratic polynominal
        
        Hg = problem.hess_vec(w, grad, sub_hess_indices);
        gHg = grad'*Hg;
        a = sigma*(norm(grad)^3);
        b = gHg;
        c = -(grad'*grad);
        [alpha_l, alpha_h] = mitternachtsformel(a, b, c);
        alpha = alpha_h;
        s = -alpha*grad;

        lambda_k = 0;

    elseif strcmp(subproblem_solver, 'exact')

        H = problem.hess(w, sub_hess_indices);
        [s, lambda_k] = exact_ARC_suproblem_solver(grad, H, sigma, exact_tol, successful_flag, lambda_k);

    elseif strcmp(subproblem_solver, 'lanczos')
        y=grad;
        grad_norm = norm(grad);
        gamma_k_next = grad_norm;
        delta = [];
        gamma = []; % save for cheaper reconstruction of Q

        dimensionality = length(w);
        if keep_Q_matrix_in_memory
            q_list = [];
        end
        
        k=0; 
        T = zeros(1,1); % Building up tri-diagonal matrix T
        
        while true
            if gamma_k_next == 0 %From T 7.5.16 u_k was the minimizer of m_k. But it was not accepted. Thus we have to be in the hard case.
                %H = hessian(w, X, Y, alpha);
                %H = hessian_f(w, new_X, new_Y, alpha);
                %[s, lambda_k] = exact_ARC_suproblem_solver(grad, H, sigma, exact_tol, successful_flag, lambda_k);
                H = problem.hess(w, sub_hess_indices);
                [s, lambda_k] = exact_ARC(grad, H, sigma, exact_tol, successful_flag, lambda_k);                
            end
            
            % a) create g
            e_1 = zeros(k+1,1);
            e_1(1) = 1.0;
            g_lanczos = grad_norm * e_1;
            
            % b) generate H
            gamma_k = gamma_k_next;
            gamma = [gamma; gamma_k];
            
            if ~(k == 0)
                q_old = q;
            end
            q = y/gamma_k;
            
            if keep_Q_matrix_in_memory
            	q_list = [q_list, q];
            end
            
            %Hq = Hv_f(w, new_X, new_Y, q, alpha); % matrix free           
            Hq = problem.hess_vec(w, q, sub_hess_indices);
            delta_k = dot(q, Hq);
            delta = [delta; delta_k];
            T_new = zeros(k + 1, k + 1);   
            if k == 0
                T(k+1,k+1) = delta_k;
                y = Hq - delta_k.* q;
            else
                T_new(1:size(T,1), 1:size(T,2)) = T;
                T_new(k+1, k+1) = delta_k;
                T_new(k, k+1) = gamma_k;
                T_new(k+1, k) = gamma_k;
                T = T_new;
                y = Hq - delta_k.* q - gamma_k.* q_old; 
            end
            
            gamma_k_next = norm(y);
            
            
            % Solve Subproblem only in each i-th Krylov space
            if mod(k, solve_each_i_th_krylov_space) ==0 || (k == dimensionality-1) || gamma_k_next == 0
                [u, lambda_k] = exact_ARC_suproblem_solver(g_lanczos, T, sigma, exact_tol, successful_flag, lambda_k);
                e_k = zeros(k+1,1);
                e_k(k+1) = 1.0;
                    if (norm(y)*abs(dot(u, e_k))) < (min(krylov_tol, norm(u)/max([1, sigma]))*grad_norm)
                        break;
                    end
            end
            
            if k == (dimensionality-1)
                fprintf('Krylov dimensionality reach full space!\n');
                break;
            end
            
            successful_flag = false;
            k = k+1;
        end
        
        % Recover Q to compute s
        n = size(grad, 1);
        Q = zeros(k + 1, n);  % <--------- since numpy is ROW MAJOR its faster to fill the transpose of Q
        y = grad;
        
        for j = 0 : k
            if keep_Q_matrix_in_memory
                Q(j+1,:) = (q_list(:,j+1))';
            else
                if ~(j == 0)
                    q_re_old = q_re;
                end
                q_re = y/gamma(j+1);
                Q(:,j+1) = q_re;
                %Hq = Hv_f(w, X, Y, q_re, alpha); % matrix free
                Hq = problem.hess_vec(w, q_re, sub_hess_indices);
                
                if j == 0
                    y = Hq - delta(j+1)*q_re;
                elseif ~(j==k)
                    y = Hq - delta(j+1)*q_re - gamma(j+1)*q_re_old;
                end
            end
        end
        s = (u'*Q)';


    else
        fptinf('unknown solver %s\n', subproblem_solver);
    end
        
end

function [s, lambda_j] = exact_ARC_suproblem_solver(grad, H, sigma, eps_exact, successful_flag, lambda_k)
    s = zeros(size(grad));
    
    % a) EV Bounds
    list_l = [];
    list_u = [];
    for i = 1 : length(H)
        l = H(i, i) - sum(abs(H(i, :))) + abs(H(i, i));
        u = H(i, i) + sum(abs(H(i, :))) - abs(H(i, i));
        list_l = [list_l l];
        list_u = [list_u u];
    end
    gershgorin_l = min(list_l);
    gershgorin_u = max(list_u);
    H_ii_min = min(diag(H));
    [h_shape,~] = size(H);
    absH = abs(H);
    H_max_norm = sqrt(h_shape.^2) * max(absH(:));
    H_fro_norm = norm(H,'fro');
    
    % b) solve quadratic equation that comes from combining rayleigh coefficients
    [lambda_l1, lambda_u1] = mitternachtsformel(1, gershgorin_l, -norm(grad)*sigma);
    [lambda_u2, lambda_l2] = mitternachtsformel(1, gershgorin_u, -norm(grad)*sigma);
    
    lambda_lower = max([0, -H_ii_min, lambda_l2]);
    lambda_upper = max([0, lambda_u1]); % 0's should not be necessary
    
    if (successful_flag == false) && (lambda_lower <= lambda_k) && (lambda_k <= lambda_upper) % reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j = lambda_k;
    else
        lambda_j = lambda_lower + (lambda_upper-lambda_lower)*rand(1,1);
    end
    
    no_of_calls = 0;
    for v = 1 : 50
        no_of_calls = no_of_calls + 1;
        lambda_plus_in_N = false;
        lambda_in_N = false;

        B = H + lambda_j * eye(size(H));
        if ((lambda_lower == 0) && (lambda_upper == 0)) || (any(grad(:)) == 0)
        	lambda_in_N = true;
        else
            try % if this succeeds lambda is in L or G.
                % 1 Factorize B
                L = chol(B);
                % 2 Solve LL^Ts=-g
                Li = inv(L);
                s = - (Li*Li')*grad;
                sn = norm(s);
               
                % 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                phi_lambda = 1.0/sn -sigma/lambda_j;
                if (abs(phi_lambda) <= eps_exact)
                    break
                end
                % 3 Solve Lw=s
                w = Li'*s;
                wn = norm(w);
                
                % Step 1: Lambda in L and thus lambda+ in L
                if phi_lambda < 0
                    % print ('lambda: ',lambda_j, ' in L')
                    [c_lo,c_hi] = mitternachtsformel((wn^2/sn^3), 1.0/sn+(wn^2/sn^3)*lambda_j, 1.0/sn*lambda_j - sigma);
                    lambda_plus = lambda_j + c_hi;
                    % lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                    lambda_j = lambda_plus;
                    
                % Step 2: Lambda in G, hard case possible
                elseif phi_lambda > 0
                    % print ('lambda: ',lambda_j, ' in G')
                    % lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                    lambda_upper = lambda_j;
                    [lo, c_hi] = mitternachtsformel((wn^2/sn^3), 1.0/sn+(wn^2/sn^3)*lambda_j, 1.0/sn*lambda_j - sigma);
                    lambda_plus = lambda_j + c_hi;
                    % Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                    if lambda_plus > 0
                        try
                            % 1 Factorize B
                            B_plus = H + lambda_plus*eye(size(H,1), size(H,2));
                            L = chol(B_plus);
                            lambda_j = lambda_plus;
                            % print ('lambda+', lambda_plus, 'in L')
                        catch
                             lambda_plus_in_N = true;
                        end
                    end
                    %Step 2b/c: else lambda+ in N, hard case possible
                    if lambda_plus <= 0 || lambda_plus_in_N == true
                        % print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower = max([lambda_lower, lambda_plus]); % reset lower safeguard
                        lambda_j = max([sqrt(lambda_lower*lambda_upper), lambda_lower+0.01*(lambda_upper-lambda_lower)]);

                         lambda_lower = single(lambda_lower);
                         lambda_upper = single(lambda_upper);
                        if lambda_lower == lambda_upper
                                lambda_j = lambda_lower; %should be redundant?
                                [ev, ew] = eig(H);
                                d = ev(:, 1);
                                dn = norm(d);
                                % note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                [tao_lower, tao_upper] = mitternachtsformel(1, 2*dot(s,d), dot(s,s)-lambda_j^2/sigma^2);
                                s = s + tao_lower * d; % both, tao_l and tao_up should give a model minimizer!
                                fprintf('hard case resolved\n') ;
                                break
                        end
                     %else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                     % lambda_in_N = True
                    end
                end
                % Step 3: Lambda in N
            catch
                lambda_in_N = true;
            end 
        end
        
        if lambda_in_N == true
            % print ('lambda: ',lambda_j, ' in N')
            lambda_lower = max([lambda_lower, lambda_j]);  % reset lower safeguard
            lambda_j = max([sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower)]);  % eq 7.3.1
            % Check Hardcase
            % if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
             lambda_lower = single(lambda_lower);
             lambda_upper = single(lambda_upper);

            if lambda_lower == lambda_upper
                lambda_j = lambda_lower; % should be redundant?
                [ev, ew] = eig(H);
                d = ev(:, 1);
                dn = norm(d);
                if any(diag(ew)) >= 0 % H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                    break;
                end
                % note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                sn = norm(s);
                [tao_lower, tao_upper] = mitternachtsformel(1, 2*dot(s,d), dot(s,s) - lambda_j^2/sigma^2);
                s = s + tao_lower * d;
                fprintf ('hard case resolved');
                break;
            end
        end
    end
   
end

% Auxiliary Functions

function [t_lower, t_upper] = mitternachtsformel(a, b, c)
    sqrt_discriminant = sqrt(b * b - 4 * a * c);
    t_lower = (-b - sqrt_discriminant) / (2 * a);
    t_upper = (-b + sqrt_discriminant) / (2 * a);
end