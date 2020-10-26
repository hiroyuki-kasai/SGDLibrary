function alpha = exact_line_search(problem, solver, p, r, y, x, options)
% Exact line search for quadratic function
%
% Reference:
%
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       solver:     GD: quadratic
%                   sub_mode    'STANDARD'  Equqtion (3.55) in Section 3.1.
%                   sub_mode    'SCALING'   
%
%       solver:     CG: quadratic
%                   sub_mode    'PRELIM'    Algorithm 5.1 in Section 5.1.
%                   sub_mode    'STANDARD'  Algorithm 5.2 in Section 5.1.
%                   sub_mode    'PRECON'    Algorithm 5.3 in Section 5.1.
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Oct. 31, 2016

    alpha = 1;

    switch solver
        case 'SD'
            grad = problem.full_grad(x);      
            
            switch problem.name()
                case 'quadratic'
                    if strcmp(options.sub_mode, 'STANDARD')
                        %alpha_old = norm(grad)^2/(2 * grad' * problem.A() * grad);
                        alpha = - grad' * p /(p' * problem.A() * p);                
                    elseif strcmp(options.sub_mode, 'SCALING')
                        S = options.S;
                        %alpha = grad' * S * grad/(2 * (grad'*S') * problem.A() * (S * grad));     
                        alpha = - p' * S * grad/((p'*S') * problem.A() * (S * p));   
                    else
                    end
                otherwise
            end
            
        case 'CG'
            
            switch problem.name()
                case 'quadratic'
                    if strcmp(options.sub_mode, 'PRELIM')
                        alpha = - r' * p /(p' * problem.A() * p);                          
                    elseif strcmp(options.sub_mode, 'STANDARD')
                        alpha = r' * r /(p' * problem.A() * p);                
                    elseif strcmp(options.sub_mode, 'PRECON')
                        alpha = r' * y /(p' * problem.A() * p);   
                    else
                    end
                otherwise
            end  
            
        case 'BFGS'
            grad = problem.full_grad(x);              
            
            switch problem.name()
                case 'quadratic'                
                    alpha = - grad' * p /(p' * problem.A() * p);                               
                otherwise
            end                
            
        otherwise
    end

end

