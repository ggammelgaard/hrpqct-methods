clc; clear; close all;

x = -10:0.1:10;

epsilon=1;%0.32;
y_heaviside = 0.5 * (1 + (2/pi)*atan(x/epsilon));
y_dirac = (1/pi)*(epsilon./(epsilon.^2 + x.^2));

%y_dirac = 5734161139222659./(18014398509481984*(x.^2 + 1));


% Globals
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
opengl software; 

% Settings
my_ylim = [-0.1 1.1];
my_lw = 3;
my_clr = '#994F88';


% Plots
f1 = figure();
p1 = plot([0 0], my_ylim, 'color','#999999');
p1.Color(4) = 0.4;
hold on
p2 = plot([min(x) max(x)], [0 0], 'color','#999999');
p2.Color(4) = 0.5;
p3 = plot(x,y_heaviside,'LineWidth',my_lw,'color',my_clr);
ylim(my_ylim);
set(gca,'FontSize',12)
set(f1,'Position',[680   558   560*0.8  420*0.8])
xlabel('$x$','Interpreter','latex')
ylabel('$\mathcal{H}(x)$','Interpreter','latex')

f2 = figure();
p1 = plot([0 0], my_ylim, 'color','#999999');
p1.Color(4) = 0.4;
hold on
p2 = plot([min(x) max(x)], [0 0], 'color','#999999');
p2.Color(4) = 0.5;
p3 = plot(x,y_dirac,'LineWidth',my_lw,'color',my_clr);
ylim(my_ylim);
set(gca,'FontSize',12)
set(f2,'Position',[680   558   560*0.8   420*0.8])
xlabel('$x$','Interpreter','latex')
ylabel('$\delta (x)$','Interpreter','latex')