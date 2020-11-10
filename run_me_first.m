% Add folders to path.

addpath(pwd);

cd sgd_solver/;
addpath(genpath(pwd));
cd ..;

cd problem/;
addpath(genpath(pwd));
cd ..;

cd tool/;
addpath(genpath(pwd));
cd ..;

cd plotter/;
addpath(genpath(pwd));
cd ..;

cd sgd_test/;
addpath(genpath(pwd));
cd ..;

% for GDLibrary
cd gd_solver/;
addpath(genpath(pwd));
cd ..;

cd gd_test/;
addpath(genpath(pwd));
cd ..;

[version, release_date] = sgdlibrary_version();
fprintf('##########################################################\n');
fprintf('###                                                    ###\n');
fprintf('###                Welcome to SGDLibrary               ###\n');
fprintf('###    (version:%s, released:%s)     ###\n', version, release_date);
fprintf('###                                                    ###\n');
fprintf('##########################################################\n');


