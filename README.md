# CI GMVC : 
----------

Authors: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) and Mitsuhiko Horie

Last page update: June 08, 2021

Latest version: 1.0.0 (see Release notes for more info) 

<br />

Introduction
----------
Multi-view data analysis has gained increasing popularity because multi-view data are frequently encountered in machine learning applications. A simple but promising approach for clustering of multi-view data is multi-view clustering (MVC), which has been developed extensively to classify given subjects into some clustered groups by learning latent common features that are shared across multi-view data. Among existing approaches, graph-based multi-view clustering (GMVC) achieves state-of-the-art performance by leveraging a shared graph matrix called the unified matrix. However, existing methods including GMVC do not explicitly address inconsistent parts of input graph matrices. Consequently, they are adversely affected by unacceptable clustering performance. 

A new algorithm called CI-GMVC is proposed as a new GMVC method that incorporates consistent and inconsistent parts lying across multiple views. This repository contains the code of CI-GMVC proposed in the following paper:
<br />


<br />

Paper
----------

M. Horie and H. Kasai, "Consistency-aware and inconsistency-aware graph-based multi-view clustering," EUSIPCO 2020. [Publisher's site](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001472.pdf), [arXiv](https://arxiv.org/abs/2011.12532)




<br />


Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script. 
|tools                  - Contains some files for execution.
|datasets               - Contains some datasets.
</pre>

<br />  

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

<br />
<br />

Execute a demo file
----------------------------
Run `demo`. 
```Matlab
%% Perform a demo.m
demo; 
```

<br />



<br />

Notes
-------
* Some parts of ci_gmvc.m are borrowed from two works below: 

    - Hao Wang, Yan Yang, Bing Liu, Hamido Fujita, "A Study of Graph-based System for Multi-view Clustering," Knowledge-Based Systems, 2019, [Code](https://github.com/cswanghao/gbs).
    - Youwei Liang, Dong Huang, and Chang-Dong Wang. Consistency Meets, "Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering," IEEE International Conference on Data Mining(ICDM), 2019, [Code](https://github.com/youweiliang/ConsistentGraphLearning).

<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) (email: hiroyuki **dot** kasai **at** waseda **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.0 (June 08, 2021)
    - Initial version.
