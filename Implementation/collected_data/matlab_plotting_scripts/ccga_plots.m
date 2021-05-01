load ..\ccga\ccga_rast.txt
load ..\ccga\ccga_schw.txt
load ..\ccga\ccga_ackl.txt
load ..\ccga\ccga_grie.txt

ccga_plot = figure();

% Rastrigin Function
subplot(2, 2, 1)
plot(ccga_rast(:,1), ccga_rast(:,2))
subtitle("Rastrigin Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Schwefel Function
subplot(2, 2, 2)
plot(ccga_schw(:,1), ccga_schw(:,2))
subtitle("Schwefel Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Griewangk Function
subplot(2, 2, 3)
plot(ccga_grie(:,1), ccga_grie(:,2))
subtitle("Griewangk Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Ackley Function
subplot(2, 2, 4)
plot(ccga_ackl(:,1), ccga_ackl(:,2))
subtitle("Ackley Function")
ylabel("Best Individual")
xlabel("Function Evaluations")