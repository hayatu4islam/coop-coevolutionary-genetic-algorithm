load ..\ga\ga_rast.txt
load ..\ga\ga_schw.txt
load ..\ga\ga_ackl.txt
load ..\ga\ga_grie.txt

load ..\ccga\ccga_rast.txt
load ..\ccga\ccga_schw.txt
load ..\ccga\ccga_ackl.txt
load ..\ccga\ccga_grie.txt

load ..\exga\exga_rast.txt
load ..\exga\exga_schw.txt
load ..\exga\exga_ackl.txt
load ..\exga\exga_grie.txt


combined_plot = figure();

% Rastrigin Function
subplot(2, 2, 1)
plot(ga_rast(:,1), ga_rast(:,2))
hold on
plot(ccga_rast(:,1), ccga_rast(:,2))
plot(exga_rast(:,1), exga_rast(:,2))
legend("GA", "CCGA", "EXGA")
subtitle("Rastrigin Function")
ylabel("Best Individual")
xlabel("Function Evaluations")
xlim([0 100000])
ylim([0 40])

% Schwefel Function
subplot(2, 2, 2)
plot(ga_schw(:,1), ga_schw(:,2))
hold on
plot(ccga_schw(:,1), ccga_schw(:,2))
plot(exga_schw(:,1), exga_schw(:,2))
legend("GA", "CCGA", "EXGA")
subtitle("Schwefel Function")
ylabel("Best Individual")
xlabel("Function Evaluations")
xlim([0 100000])
ylim([0 400])

% Griewangk Function
subplot(2, 2, 3)
plot(ga_grie(:,1), ga_grie(:,2))
hold on
plot(ccga_grie(:,1), ccga_grie(:,2))
plot(exga_grie(:,1), exga_grie(:,2))
legend("GA", "CCGA", "EXGA")
subtitle("Griewangk Function")
ylabel("Best Individual")
xlabel("Function Evaluations")
xlim([0 100000])
ylim([0 8])

% Ackley Function
subplot(2, 2, 4)
plot(ga_ackl(:,1), ga_ackl(:,2))
hold on
plot(ccga_ackl(:,1), ccga_ackl(:,2))
plot(exga_ackl(:,1), exga_ackl(:,2))
legend("GA", "CCGA", "EXGA")
subtitle("Ackley Function")
ylabel("Best Individual")
xlabel("Function Evaluations")
xlim([0 100000])
ylim([0 16])
