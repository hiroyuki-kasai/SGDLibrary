function [ algs ] = gd_solver_list(category)
% Return list of solvers.
%
% Inputs:
%       category    category to be returned. 
% Output:
%       algs        list of solvers in the category
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Nov. 02, 2016


    % supported algorithms by GDLibrary
    gd_algs = {'GD-STD','GD-BKT','GD-EXACT','GD-WOLFE','GD-SCALE-EXACT'};
    newton_algs = {'Newton-STD','Newton-DAMP','Newton-CHOLESKY'};
    cg_algs = {'CG-PRELIM','CG-BKT','CG-EXACT','CG-PRECON-EXACT'};
    ncg_algs = {'NCG-BTK','NCG-WOLFE'};
    bfgs_algs = {'BFGS-B-EXACT', 'BFGS-B-BKT','BFGS-H-EXACT','BFGS-H-BKT', ...
                    'DAMPED-BFGS-BKT','DAMPED-BFGS-EXACT', ...
                    'L-BFGS-BKT','L-BFGS-EXACT','L-BFGS-WOLFE'}; 
    other_algs = {'BB'};
    
    linesearch_algs = {'GD-BKT','GD-WOLFE','NCG-BTK','NCG-WOLFE','BFGS-B-BKT','BFGS-H-BKT','DAMPED-BFGS-BKT','L-BFGS-BKT','L-BFGS-WOLFE'};
    exact_algs = {'GD-EXACT','GD-SCALE-EXACT','CG-EXACT','CG-PRECON-EXACT','BFGS-B-EXACT','BFGS-H-EXACT','DAMPED-BFGS-EXACT','L-BFGS-EXACT'};    
    
    switch category
        case 'GD'
            algs = gd_algs;
        case 'CG'
            algs = cg_algs;  
        case 'NCG'
            algs = ncg_algs;              
        case 'Newton'
            algs = newton_algs;   
        case 'BFGS'
            algs = bfgs_algs;   
        case 'LS'            
            algs = linesearch_algs;      
        case 'EXACT'            
            algs = exact_algs;               
        case 'ALL'
            algs = [gd_algs, newton_algs, cg_algs, bfgs_algs, other_algs];
        otherwise
    end
    
end
