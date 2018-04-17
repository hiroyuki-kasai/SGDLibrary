function [s, lambda_k] = tr_subsolver(problem, w, grad, tr_radius, sub_hess_indices, ...
                                                successful_flag, lambda_k, subproblem_solver,...
                                                exact_tol, krylov_tol)
    if strcmp(subproblem_solver, 'cauchy_point')
        %Hg = Hv_f(w, new_X, new_Y, grad,alpha);
        Hg = problem.hess_vec(w, grad, sub_hess_indices);
        gBg = grad'*Hg;
        tau = 1;
        if gBg > 0  % if model is convex quadratic the unconstrained minimizer may be inside the TR
            tau = min([(norm(grad))^3 / (tr_radius * gBg), 1]);
        end
        pc = - tau * tr_radius * (grad./norm(grad));
        s = pc;
        lamda_k = 0;
    elseif strcmp(subproblem_solver, 'dog_leg')
        %H = hessian_f(w, new_X, new_Y, alpha);
        H = problem.hess(w, sub_hess_indices);
        gBg = grad'*(H'*grad);
        if gBg <= 0
            error('dog_leg requires H to be positive definite in all steps!') ;
        end

        % Compute the Newton Point and return it if inside the TR
        L = chol(H,'lower');
        y = L\grad;
        pn = -L'\y;
%         cholesky_B = chol(H,'lower');
%         pn = - (cholesky_B\grad);
        if (norm(pn) < tr_radius)
            s = pn;
            lambda_k = 0;
            return;
        end

        % Compute the 'unconstrained Cauchy Point'
        pc = -((grad'*grad)/gBg) * grad;
        pc_norm = norm(pc);

        % if it is outside the TR, return the point where the path intersects the boundary
        if pc_norm >= tr_radius
            p_boundary = pc * (tr_radius / pc_norm);
            s = p_boundary;
            lambda_k = 0;
            return;
        end


        % else, give intersection of path from pc to pn with tr_radius.
        [t_lower, t_upper] = solve_quadratic_equation(pc, pn, tr_radius);
        p_boundary = pc + t_upper * (pn - pc);
        s = p_boundary;
        lambda_k = 0;

    elseif strcmp(subproblem_solver, 'cg')
        grad_norm = norm(grad);
        p_start = zeros(size(grad,1),size(grad,2));

        if grad_norm < min([sqrt(norm(grad)) * norm(grad), krylov_tol])
            s = p_start;
            lambda_k = 0;
            return;
        end

        % initialize
        z = p_start;
        r = grad;
        d = -r;
        k = 0;
          
        while true
            %Bd = Hv_f(w, new_X, new_Y, d, alpha);
            Bd = problem.hess_vec(w, d, sub_hess_indices);
            dBd = d'*Bd;
            % terminate when encountering a direction of negative curvature with lowest boundary point along current search direction
            if dBd <= 0
                [t_lower, t_upper] = solve_quadratic_equation(z, d, tr_radius);
                p_low = z + t_lower * d;
                p_up = z + t_upper * d;
                %m_p_low = loss_f(w + p_low, X, Y, alpha) + grad'*p_low + 0.5 * p_low'*(H'*p_low);
                %m_p_up = loss_f(w + p_up, X, Y, alpha) + grad'*p_up + 0.5 * p_up'*(H'*p_up);
                
                H = problem.hess(w, sub_hess_indices); % Added by HK (Need to be checked.)
                m_p_low = problem.cost(w+p_low) + grad'*p_low + 0.5 * p_low'*(H'*p_low);
                m_p_up = problem.cost(w+p_up) + grad'*p_up + 0.5 * p_up'*(H'*p_up);                
                problem.cost(w);
                if m_p_low < m_p_up
                    s = p_low;
                    lambda_k = 0;
                    return;
                else
                    s = p_up;
                    lambda_k = 0;
                    return;
                end
            end


            alpha = (r'*r) / dBd;
            z_next = z + alpha * d;
            % terminate if z_next violates TR bound
            if norm(z_next) >= tr_radius
                % return intersect of current search direction w/ boud
                [t_lower, t_upper] = solve_quadratic_equation(z, d, tr_radius);
                s = z + t_upper * d;
                lambda_k = 0;
                return;
            end
            
            r_next = r + alpha * Bd;
            if norm(r_next) < min([sqrt(norm(grad)) * norm(grad),krylov_tol])
                s = z_next;
                lambda_k = 0;
                return;
            end

            beta_next = (r_next'*r_next) / (r'*r);
            d_next = -r_next + beta_next * d;
            % update iterates
            z = z_next;
            r = r_next;
            d = d_next;
            k = k + 1;
        end
    elseif strcmp(subproblem_solver, 'GLTR')
        g_norm = norm(grad);
        s = zeros(size(grad,1), size(grad,2));
   
        if g_norm == 0
            % escape along the direction of the leftmost eigenvector as far as tr_radius permits
            fprintf('zero gradient encountered');
            %H = hessian_f(w, new_X, new_Y, alpha);
            H = problem.hess(w, sub_hess_indices);
            [s, lambda_k] = exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol, successful_flag, lambda_k);
        else
            % initialize
            g = grad;
            p = -g;
            gamma = g_norm;
            T = zeros(1, 1);
            alpha_k = [];
            beta_k = [];
            interior_flag = true;
            k = 0;
            
            while true
                %Hp = Hv_f(w, new_X, new_Y, p, alpha);
                Hp = problem.hess_vec(w, p, sub_hess_indices);
                pHp = p'*Hp;
                alpha = g'*g / pHp;
               
                alpha_k = [alpha_k; alpha];
                
                %Lanczos Step 1: Build up subspace 
                % a) Create g_lanczos = gamma*e_1
                
                e_1 = zeros(k + 1, 1);
                e_1(1) = 1.0;
                g_lanczos = gamma .* e_1;
                
                % b) Create T for Lanczos Model 
                T_new = zeros(k + 1, k + 1);
                if k == 0
                    T(k+1, k+1) = 1.0/alpha;
                    T_new(1:k+1, 1:k+1) = T;
                else
                    T_new(1:size(T,1), 1:size(T,2)) = T;
                    T_new(k+1, k+1) = 1.0 / alpha + beta_/ alpha_k(k);
                    T_new(k, k+1) = sqrt(beta_) / abs(alpha_k(k));
                    T_new(k+1, k) = sqrt(beta_) / abs(alpha_k(k));
                    T = T_new; 
                end
                
                if (interior_flag == true && alpha < 0) || norm(s + alpha * p) >= tr_radius
                    interior_flag = false;
                end
                
                if interior_flag == true
                    s = s + alpha * p;
                else
                    % Lanczos Step 2: solve problem in subspace
                    [h, lambda_k] = exact_TR_suproblem_solver(g_lanczos, T, tr_radius, exact_tol, successful_flag, lambda_k);
                end
                g_next = g + alpha * Hp;
                
                % test for convergence
                e_k = zeros(k + 1,1);
                e_k(k+1) = 1.0;
                
                if interior_flag == true && norm(g_next) < min([sqrt(norm(grad)) * norm(grad), krylov_tol])
                    break;
                end
                if interior_flag == false && norm(g_next) * abs(h'*e_k) < min([sqrt(norm(grad)) * norm(grad), krylov_tol]) 
                    break;
                end
                
                if k == problem.d
                    fprintf('Krylov dimensionality reach full space! Breaking out..\n');
                    break;
                end
                beta_ = (g_next'*g_next) / (g'*g);
                beta_k = [beta_k; beta_];
                p = -g_next + beta_ * p;
                g = g_next;
                k = k + 1;
            end
            
            if interior_flag == false
                % Recover Q by building up the lanczos space, TBD: keep storable Qs in memory
                n = size(grad,1);
                Q1 = zeros(n, k + 1);
                
                g = grad;
                p = -g;
                
                for j = 0 : k
                    gn = norm(g);
                    if j == 0
                        sigma = 1;
                    else
                        sigma = -sign(alpha_k(j)) * sigma;
                    end
                    Q1(:, j+1) = sigma * g / gn;
                    

                    if ~ (j == k)
                        %Hp = Hv_f(w, new_X, new_Y, p, alpha);
                        Hp = problem.hess_vec(w, p, sub_hess_indices);
                        g = g + alpha_k(j+1) * Hp;
                        p = -g + beta_k(j+1) * p;
                    end
                end
                
                % compute final step in R^n
                s = Q1*h;
            end
        end
    elseif strcmp(subproblem_solver, 'exact')
        %H = hessian_f(w, new_X, new_Y, alpha);
        Hw = problem.hess(w, sub_hess_indices);
        [s, lambda_k] = exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol, successful_flag, lambda_k);
    else
    	error('solver unknown\n');
    end
end

function [s, lambda_j] = exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol, successful_flag, lambda_k)

    s = zeros(size(grad,1),size(grad,2));
    % Step 0: initialize safeguards
    H_ii_min = min(diag(H));
    absH = abs(H);
    H_max_norm = sqrt((size(H,1))^2) * max(absH(:));
    H_fro_norm = norm(H, 'fro');
    list_l = [];
    list_u = [];
    for i = 1 : length(H)
        l = H(i, i) + sum(abs(H(i, :))) - abs(H(i, i));
        u = -H(i, i) + sum(abs(H(i, :))) - abs(H(i, i));
        list_l = [list_l l];
        list_u = [list_u u];
    end
    gerschgorin_l = max(list_l);
    gerschgorin_u = max(list_u);

    lambda_lower = max([0, -H_ii_min, norm(grad) / tr_radius - min([H_fro_norm, H_max_norm, gerschgorin_l])]);
    lambda_upper = max([0, norm(grad) / tr_radius + min([H_fro_norm, H_max_norm, gerschgorin_u])]);

    if successful_flag == false && (lambda_lower <= lambda_k) && (lambda_k <= lambda_upper) % reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j = lambda_k;
    elseif lambda_lower == 0  % allow for fast convergence in case of inner solution
        lambda_j = lambda_lower;
    else
        lambda_j = lambda_lower + (lambda_upper-lambda_lower)*rand(1,1);
    end

    i = 0;
    % Root Finding
    while true
        i = i + 1;
        lambda_in_N = false;
        lambda_plus_in_N = false;
        B = H + lambda_j * eye(size(H,1), size(H,2));
        try
            % 1 Factorize B
            L = chol(B, 'lower');
            % 2 Solve LL^Ts=-g
            Li = inv(L);
            s = - (Li'*Li)*grad;
            sn = norm(s);
            % 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
            phi_lambda = 1.0 / sn - 1.0 / tr_radius;
            %if (abs(sn - tr_radius) <= exact_tol * tr_radius):
            if (abs(phi_lambda)<=exact_tol) %
                break;
            end
            
            % 3 Solve Lw=s
                w = Li*s;
                wn = norm(w);
            
            % Step 1: Lambda in L
            if lambda_j > 0 && (phi_lambda) < 0
                % print ('lambda: ',lambda_j, ' in L')
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * ((sn^2) / (wn^2));
                lambda_j = lambda_plus;
                
            % Step 2: Lambda in G    (sn<tr_radius)
            elseif (phi_lambda) > 0 && lambda_j > 0 && any(grad ~= 0) % TBD: remove grad
                % print ('lambda: ',lambda_j, ' in G')
                lambda_upper = lambda_j;
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * ((sn^2) / (wn^2));
                
                % Step 2a: If factorization succeeds: lambda_plus in L
                if lambda_plus > 0
                    try
                        % 1 Factorize B
                        B_plus = H + lambda_plus * eye(size(H,1), size(H,2));
                        L = chol(B_plus, 'lower');
                        lambda_j = lambda_plus;
                        % print ('lambda+', lambda_plus, 'in L')
                    catch 
                        lambda_plus_in_N = true;
                    end
                end
                
                % Step 2b/c: If not: Lambda_plus in N
                if lambda_plus <= 0 || lambda_plus_in_N == true
                    % 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                    try
                        U = chol(H, 'upper');
                        H_pd = true;
                    catch 
                        H_pd = false;
                    end
                    
                    if lambda_lower == 0 && H_pd == true && phi_lambda >= 0 %cannot happen in ARC!
                        lambda_j = 0;
                        % print ('inner solution found');
                        break;
                    % 2. Else, choose a lambda within the safeguard interval
                    else
                        % print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower = max([lambda_lower, lambda_plus]);  % reset lower safeguard
                        lambda_j = max([sqrt(lambda_lower * lambda_upper),lambda_lower + 0.01 * (lambda_upper - lambda_lower)]);
                        lambda_upper = single(lambda_upper); 
                        
                        if lambda_lower == lambda_upper
                            lambda_j = lambda_lower;
                            % Hard case
                            [ev, ew] = eig(H);
                            d = ev(:, 1);
                            dn = norm(d);
                            assert((ew == -lambda_j), 'Ackward: in hard case but lambda_j != -lambda_1');
                            [tao_lower, tao_upper] = mitternachtsformel(1, 2*(s'*d), (s'*s)-tr_radius^2);
                            s = s + tao_lower * d;
                            fprintf('hard case resolved inside');
                        end
                    end
                end
            
            elseif (phi_lambda) == 0
                break;
            else % TBD:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                lambda_in_N = true;
            end   
        % Step 3: Lambda in N
        catch
            lambda_in_N = true;
        end
    
        if lambda_in_N == true
            % print ('lambda: ',lambda_j, ' in N')
            try
                U = chol(H, 'upper');
                H_pd = true;
            catch 
                H_pd = false;
            end
            
            % 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
            if lambda_lower == 0 && H_pd == true && phi_lambda >= 0
                lambda_j = 0;
                %print ('inner solution found')
                break;
            % 2. Else, choose a lambda within the safeguard interval
            else
                lambda_lower = max([lambda_lower, lambda_j]);  % reset lower safeguard
                lambda_j = max([sqrt(lambda_lower * lambda_upper),lambda_lower + 0.01 * (lambda_upper - lambda_lower)]);  % eq 7.3.14
                lambda_upper = single(lambda_upper);
                % Check for Hard Case:
                if lambda_lower == lambda_upper
                    lambda_j = lambda_lower;
                    [ev, ew] = eig(H);
                    d = ev(:, 1);
                    dn = norm(d);
                    assert((ew == -lambda_j), 'Ackward: in hard case but lambda_j != -lambda_1');
                    [tao_lower, tao_upper] = mitternachtsformel(1, 2*(s'*d), (s'*s)-tr_radius^2);
                    s = s + tao_lower * d;

                    fprintf('hard case resolved outside');
                end
            end
        end
    end
    
    % compute final step
    B = H + lambda_j * eye(size(H,1), size(H,2));
    % 1 Factorize B
    L = chol(B, 'lower');
    % 2 Solve LL^Ts=-g
    Li = inv(L);
    s = - Li'*Li*grad;
    %print (i,' exact solver iterations')

end


% Auxiliary Functions
function [t_lower, t_upper] = mitternachtsformel(a, b, c)
    sqrt_discriminant = sqrt(b * b - 4 * a * c);
    t_lower = (-b - sqrt_discriminant) / (2 * a);
    t_upper = (-b + sqrt_discriminant) / (2 * a);
end

function [t_lower, t_upper] = solve_quadratic_equation(pc, pn, tr_radius)
    % solves ax^2+bx+c=0
    a = (pn - pc)'*(pn - pc);
    b = 2 * (pc'*(pn - pc));
    c = (pc'*pc) - tr_radius^2;
    sqrt_discriminant = sqrt(b * b - 4 * a * c);
    t_lower = (-b - sqrt_discriminant) / (2 * a);
    t_upper = (-b + sqrt_discriminant) / (2 * a);
end
