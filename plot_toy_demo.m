clc
clear all

ite = 10;
AT = 'PGDAT_KL/';
%AT = 'PGDATconsistent_KL/';
%AT = 'Standard_KL/';

% x_test=importdata([AT,'x_test_',num2str(ite),'.txt']);
% y_test=importdata([AT,'y_test_',num2str(ite),'.txt']);
% pre_test=importdata([AT,'pre_test_',num2str(ite),'.txt']);

x_test=importdata([AT,'x_test.txt']);
y_test=importdata([AT,'y_test.txt']);
pre_test=importdata([AT,'pre_test.txt']);

color_1 = [0 139 139]/255;
color_2 = [0.8500 0.3250 0.0980];
Msize = 80;

hold on
scatter(x_test, y_test(:,1), Msize, 'filled','MarkerEdgeColor',color_1,'MarkerFaceColor',color_1)
scatter(x_test, pre_test(:,1),Msize, 'filled','MarkerEdgeColor',color_2,'MarkerFaceColor',color_2)
grid on
ylim([0 1])
yticks(0:0.1:1)
xlim([-5 8])
xticks(-5:1:8)
set(gca, 'FontName', 'Times New Roman','FontSize', 23);

% y_ref=0:0.01:1;
% ref1=ones(1,101)*1;
% plot(ref1, y_ref,'k--','LineWidth',3)

h(1) = plot(nan, nan, 'o', 'MarkerSize', 15, 'MarkerEdgeColor',color_1,'MarkerFaceColor',color_1,'LineWidth',2.5, 'DisplayName', ' Ground truth');
h(2) = plot(nan, nan, 'o', 'MarkerSize', 15, 'MarkerEdgeColor',color_2,'MarkerFaceColor',color_2,'LineWidth',2.5, 'DisplayName', ' Learned by PGD-AT');
%legend(h)

% xl = xlabel('$X$', 'Interpreter','latex');
% set(xl,'FontSize',28,'FontName', 'Times New Roman');
% yl = ylabel('$P(Y=0|X)$', 'Interpreter','latex');
% set(yl,'FontSize',28,'FontName', 'Times New Roman');
