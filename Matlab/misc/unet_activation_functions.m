clc; clear; close all;

x = (-5:0.1:5);
y_sigmoid = 1 ./ (1+exp(-x));
y_relu = max(0,x);
a = 0.3;
y_lrelu = max(a*x,x);

% Globals
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
opengl software; 

% Settings
my_ylim = [-5 5];
my_xlim = [-5 5];
my_XTick = -10:5:10;%-10:2.5:10;
my_lw = 3;
my_clr = '#994F88';
img_dims = [680   558   280   270];%[680   558   560*0.5   420*0.6];

% Plots
f1 = figure(); plot(x,y_sigmoid,'LineWidth',my_lw,'color',my_clr);
grid on
xlim(my_xlim);
ylim([0 1]);
ax = gca;
ax.YTick  = -10:0.5:10; %-10:0.25:10;
ax.XTick  = my_XTick;
set(gca,'FontSize',11)
set(f1,'Position',img_dims)
xlabel('$x$','Interpreter','latex')
ylabel('$f(x)$','Interpreter','latex')

f1 = figure(); plot(x,y_relu, 'LineWidth',my_lw,'color',my_clr);
grid on
xlim(my_xlim);
ylim(my_ylim);
ax = gca;
ax.YTick  = my_XTick;
ax.XTick  = my_XTick;
set(gca,'FontSize',10)
set(f1,'Position',img_dims)
xlabel('$x$','Interpreter','latex')
ylabel('$f(x)$','Interpreter','latex')

f1 = figure(); plot(x,y_lrelu, 'LineWidth',my_lw,'color',my_clr);
grid on
xlim(my_xlim);
ylim(my_ylim);
ax = gca;
ax.YTick  = my_XTick;
ax.XTick  = my_XTick;
set(gca,'FontSize',10)
set(f1,'Position',img_dims)
xlabel('$x$','Interpreter','latex')
ylabel('$f(x)$','Interpreter','latex')

