% Extract data from the first table (logparallel)
x1 = logparallel.VarName1; % Index column for logparallel
y1 = logparallel.VarName2; % Execution time values for logparallel

% Extract data from the second table (logsequential)
x2 = logsequential.VarName1; % Index column for logsequential
y2 = logsequential.VarName2; % Execution time values for logsequential

% Create the plot with both data series
figure;
plot(x1, y1, '-o', 'DisplayName', 'Parallel Code Execution'); % First data series
hold on;
plot(x2, y2, '-x', 'DisplayName', 'Sequential Code Execution'); % Second data series
hold off;

% Add labels and title
xlabel('Index of execution test');
ylabel('Execution Time (Clock Cycles)');
title('Parallel and Sequential Code Execution');
legend show; % Show the legend
grid on;

%second figure
% Estrai i dati dalla tabella SpeedUp
x = SpeedUp.VarName1; % colonna degli indici per SpeedUp
y = SpeedUp.VarName2; % colonna dei valori di speedup

% Calcola la media dei valori di speedup
mean_y = mean(y);

% Crea il grafico dello Speedup
figure;
plot(x, y, '-o'); % '-o' per linee con marker sui punti
hold on;

% Disegna la linea della media
yline(mean_y, '--r', 'Average', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

% Disegna la linea a y=1
yline(1, '--k', 'y=1', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

% Imposta i dettagli del grafico
xlabel('Index of execution test');
ylabel('Speedup');
title('Speedup of Code Execution');
grid on;
ylim([0, max(y)*1.1]); % Imposta il limite inferiore dell'asse y a 0 e superiore leggermente sopra il massimo valore di y

hold off;
