# SGDLibrary : Stochastic Gradient Descent Library in MATLAB
----------

Authors: [Hiroyuki Kasai](http://www.kasailab.com/)

Last Page Update: Oct. 27, 2016

Latest Library Version: 1.0.0 (see Release Notes for more info)

Introduction
----------
The SGDLibrary is a **pure-Matlab** library of a collection of stochastic optimization algorithms. This solves an unconstrained minimization problem of the form, min f(x) = sum_i f_i(x).



List of the algorithms available in SGDLibrary
---------
- **SGD** (stochastic gradient descent)
    - H. Robbins and S. Monro, "[A stochastic approximation method](https://www.jstor.org/stable/pdf/2236626.pdf)," The annals of mathematical statistics, vol. 22, no. 3, pp. 400-407, 1951.
    - L. Bottou, "[Online learning and stochastic approximations](http://leon.bottou.org/publications/pdf/online-1998.pdf)," Edited by David Saad, Cambridge University Press, Cambridge, UK, 1998.
- **Variance reduction variants**
    - SVRG (stochastic variance reduced gradient)
        - R. Johnson and T. Zhang, "[Accelerating sstochastic gradient descent using predictive variance reduction](http://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)," NIPS, 2013.
    - SAG (stochastic average gradient)
        - N. L. Roux, M. Schmidt, and F. R. Bach, "[A stochastic gradient method with an exponential convergence rate for finite training sets](https://papers.nips.cc/paper/4633-a-stochastic-gradient-method-with-an-exponential-convergence-_rate-for-finite-training-sets.pdf)," NIPS, 2012.
    - SAGA
        - A. Defazio, F. Bach, and S. Lacoste-Julien, "[SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives](https://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf),", NIPS, 2014.
- **Quasi-Newton variants**
    - SQN (stochastic quasi-Newton)
        - R. H. Byrd, ,S. L. Hansen J. Nocedal, and Y. Singer, "[A stochastic quasi-Newton method 
for large-scale optimization](http://epubs.siam.org/doi/abs/10.1137/140954362?journalCode=sjope8)," SIAM Journal on Optimization, vol. 26, Issue 2, pp. 1008-1031, 2016.
    - SVRG SQN (denoted as "Stochastic L-BFGS" or "slbfgs" in the paper below.)
        - P. Moritz, R. Nishihara and M. I. Jordan, "[A linearly-convergent stochastic L-BFGS Algorithm](http://www.jmlr.org/proceedings/papers/v51/moritz16.html)," International Conference on Artificial Intelligence and Statistics (AISTATS), pp.249-258, 2016.
    - SVRG LBFGS (denoted as "SVRG+II: LBFGS" in the paper below.)
        - R. Kolte, M. Erdogdu and A. Ozgur, "[Accelerating SVRG via second-order information](http://www.opt-ml.org/papers/OPT2015_paper_41.pdf)," OPT2015, 2015.
    - oBFGS-Inf (Online BFGS, Infinite memory)
        - N. N. Schraudolph, J. Yu and Simon Gunter, "[A stochastic quasi-Newton method for online convex optimization
](http://www.jmlr.org/proceedings/papers/v2/schraudolph07a/schraudolph07a.pdf)," 
International Conference on Artificial Intelligence and Statistics (AISTATS), pp.436-443, Journal of Machine Learning Research, 2007.
    - oLBFGS-Lim (Online BFGS, Limited memory)
        - N. N. Schraudolph, J. Yu and S. Gunter, "[A stochastic quasi-Newton method for online convex optimization
](http://www.jmlr.org/proceedings/papers/v2/schraudolph07a/schraudolph07a.pdf)," 
International Conference on Artificial Intelligence and Statistics (AISTATS), pp.436-443, Journal of Machine Learning Research, 2007.

        - A. Mokhtari and A. Ribeiro, "[Global convergence of online limited memory BFGS](www.jmlr.org/papers/volume16/mokhtari15a/mokhtari15a.pdf )," Journal of Machine Learning Research, 16, pp. 3151-3181, 2015.
    - Reg-oBFGS-Inf (Regularized oBFGS, Infinite memory) (denoted as "RES" in the paper below.)
        - A. Mokhtari and A. Ribeiro, "[RES: Regularized stochastic BFGS algorithm](http://ieeexplore.ieee.org/document/6899692/)," IEEE Transactions on Signal Processing, vol. 62, no. 23, pp. 6089-6104, Dec., 2014.
    - Damp-oBFGS-Inf (Regularized damped oBFGS, Infinite memory) (denoted as "SDBFGS" in the paper below.)
        - X. Wang, S. Ma, D. Goldfarb and W. Liu, "[Stochastic quasi-Newton methods for nonconvex stochastic 
optimization](https://arxiv.org/pdf/1607.01231v3.pdf),"  arXiv preprint arXiv:1607.01231, 2016.
- **Adagrad variants**
    - AdaGrad (Adaptive gradient algorithm)
        - J. Duchi, E. Hazan and Y. Singer, "[Adaptive subgradient methods for online learning and stochastic optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)," Journal of Machine Learning Research, 12, pp. 2121-2159, 2011.
    - AdaDelta
        - M. D.Zeiler, "[AdaDelta: An adaptive learning rate method](http://arxiv.org/abs/1212.5701)," arXiv preprint arXiv:1212.5701, 2012.
    - RMSProp
        - T. Tieleman and G. Hinton, "Lecture 6.5 - RMSProp", COURSERA: Neural Networks for Machine Learning, Technical report, 2012.
    - Adam
        - D. Kingma and J. Ba, "[Adam: A method for stochastic optimization](http://arxiv.org/pdf/1412.6980.pdf)," International Conference for Learning Representation (ICLR), 2015.
    - AdaMax
        - D. Kingma and J. Ba, "[Adam: A method for stochastic optimization](http://arxiv.org/pdf/1412.6980.pdf)," International Conference for Learning Representation (ICLR), 2015.

Algorithm configurations
---------

[](
|Algorithm name in example codes| module | options.sub_mode | other options |
|---|---|---|
|SGD|sgd|---|---|
|SVRG|svrg|---|---|
|SAG|sag|'SAG'|---|
|SAGA|sag|'SAGA'|---|
|SQN|slbfgs|'SQN'|---|
|SVRG-SQN|slbfgs|'SVRG-SQN'|---|
|SVRG-LBFGS|slbfgs|'SVRG-LBFGS'|---|
|oBFGS-Inf|obfgs|'Inf-mem'|---|
|oLBFGS-Lim|obfgs|'Lim-mem'|---|
|Reg-oBFGS-Inf|obfgs|'Inf-mem'|regularized=true|
|Damp-oBFGS-Inf|obfgs|'Inf-mem'|regularized=true & damped=true|
|AdaGrad|adagrad|'AdaGrad'|---|
|RMSProp|adagrad|'RMSProp'|---|
|AdaDelta|adagrad|'AdaDelta'|---|
|Adam|adam|'Adam'|---|
|AdaMax|adam|'AdaMax'|---|)

<img src="https://dl.dropboxusercontent.com/u/869853/github/SGDLibrary/images/algorithm_table.png" width="900">

- Note that other algorithms could be configurable by selecting other combinations of sub_mode and options. 



Supported problems
---------
* Multidimensional linear regression
* Linear SVM
* Logistic regression
* Softmax classification (multinomial logistic regression)
    - Note that softmax classification problem does not support Hessian-vector product type algorithms, i.e., SQN, SVRG-SQN and SVRG-LBFGS.

Folders and files
---------

- run_me_first.m
    - The scipt that you need to run first.

- demo.m
    - A demonstration scipt to check and understand this package easily. 
                      
- solvers
    - Contains various stochastic optimization algorithms.

- problems
    - Condins definition files to be solved.

- examples
    - Some helpful test sample scipts to use this package.

- plotting
    - Contains plotting tools to show convergence results and various plots.
                  
- tool
    - Some utility tools for this project.
                  
                              

First to do
----------------------------
Run the setup script "**run_me_first.m**" for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

Usage example (logistic regression problem)
----------------------------
Now, you just execute "**demo.m**" for demonstration of this package.
```Matlab
%% Execute the demonstration script
demo; 
```

The "**demo.m**" file contains below.
```Matlab
%% generate synthtic data        
% set number of dimensions
d = 3;
% set number of samples    
n = 100;
% generate data
data = logistic_regression_data_generator(n, d);
% set train data
x_train = data.x_train;
y_train = data.y_train;  
% set test data
x_test = data.x_test;
y_test = data.y_test;            
% set lambda 
lambda = 0.1;

%% define problem definitions
problem = logistic_regression(x_train, y_train, x_test, y_test, lambda); 


%% calculate solution 
w_star = problem.calc_solution(problem, 10000, 0.01);


%% general options for optimization algorithms   
% generate initial point
options.w_init = randn(d,1);
% set iteration optimality gap tolerance
options.tol_optgap = -Inf;
% set max epoch
options.max_epoch = 100;
% set verbose mode
options.verbose = true;
% set regularization parameter    
options.lambda = lambda;
% set solution
options.f_sol = problem.cost(w_star);
% set batch sizse    
options.batch_size = 10;
% set stepsize algorithm and stepsize
options.step_alg = 'fix';
options.step = 0.005;     



%% perform algorithms SGD
[w_sgd, info_list_sgd] = sgd(problem, options);  

% predict
y_pred_sgd = problem.prediction(w_sgd);
% calculate accuracy
accuracy_sgd = problem.accuracy(y_pred_sgd); 
fprintf('Classificaiton accuracy: %s: %.4f\n', 'SGD', accuracy_sgd);

% convert from {1,-1} to {1,2}
y_pred_sgd(y_pred_sgd==-1) = 2;
y_pred_sgd(y_pred_sgd==1) = 1;



%% perform algorithms SVRG
[w_svrg, info_list_svrg] = svrg(problem, options);  

% predict
y_pred_svrg = problem.prediction(w_svrg);

% calculate accuracy
accuracy_svrg = problem.accuracy(y_pred_svrg); 
fprintf('Classificaiton accuracy: %s: %.4f\n', 'SVRG', accuracy_svrg);

% convert from {1,-1} to {1,2}
y_pred_svrg(y_pred_svrg==-1) = 2;
y_pred_svrg(y_pred_svrg==1) = 1;



%% plot all

% display cost vs grads
display_graph('cost', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_list_sgd, info_list_svrg});

% display optimality gap vs grads
display_graph('optimality_gap', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_list_sgd, info_list_svrg});

% convert from {1,-1} to {1,2}
y_train(y_train==-1) = 2;
y_train(y_train==1) = 1;
y_test(y_test==-1) = 2;
y_test(y_test==1) = 1;  

% display classification results    
display_classification_result(problem, {'SGD', 'SVRG'}, {w_sgd, w_svrg}, ...
                                {y_pred_sgd, y_pred_svrg}, {accuracy_sgd, accuracy_svrg}, ...
                                x_train, y_train, x_test, y_test);    


```

* Output results 

<img src="https://dl.dropboxusercontent.com/u/869853/github/SGDLibrary/images/log_reg_results.png" width="900">
<br /><br />


Example results of other problems
----------------------------

- Linear regression problem  

<img src="https://dl.dropboxusercontent.com/u/869853/github/SGDLibrary/images/linear_reg_results.png" width="900">

- Softmax classifier problem

<img src="https://dl.dropboxusercontent.com/u/869853/github/SGDLibrary/images/soft_class_results.png" width="900">

- Linear SVM problem

<img src="https://dl.dropboxusercontent.com/u/869853/github/SGDLibrary/images/linear_svm_results.png" width="900">
<br /><br />

License
-------
The SGDLibrary is free and open source for academic/research purposes (non-commercial).


Notes
-------
- As always, parameters such as the stepsize should be configured properly in eash algorithm and each problem. 
- Softmax classification problem does not support "Hessian-vector product" type algorithms, i.e., SQN, SVRG-SQN and SVRG-LBFGS.


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://www.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

Release Notes
--------------

* Version 1.0.0 (Oct. 28, 2016): First version.
