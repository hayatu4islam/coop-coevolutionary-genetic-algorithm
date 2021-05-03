load ..\ga\ga_rast.txt
load ..\ga\ga_schw.txt
load ..\ga\ga_ackl.txt
load ..\ga\ga_grie.txt

load ..\ccga\ccga_rast.txt
load ..\ccga\ccga_schw.txt
load ..\ccga\ccga_ackl.txt
load ..\ccga\ccga_grie.txt

load ..\exga_1\exga_1_rast.txt
load ..\exga_1\exga_1_schw.txt
load ..\exga_1\exga_1_ackl.txt
load ..\exga_1\exga_1_grie.txt

load ..\exga_2\exga_2_rast.txt
load ..\exga_2\exga_2_schw.txt
load ..\exga_2\exga_2_ackl.txt
load ..\exga_2\exga_2_grie.txt


combined_plot = figure();

% Rastrigin Function
subplot(2, 2, 1)
plot(ga_rast(:,1), ga_rast(:,2))
hold on
plot(ccga_rast(:,1), ccga_rast(:,2))
plot(exga_1_rast(:,1), exga_1_rast(:,2))
plot(exga_2_rast(:,1), exga_2_rast(:,2))
legend("GA", "CCGA", "EXGA_1", "EXGA_2")
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
plot(exga_1_schw(:,1), exga_1_schw(:,2))
plot(exga_2_schw(:,1), exga_2_schw(:,2))
legend("GA", "CCGA", "EXGA_1", "EXGA_2")
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
plot(exga_1_grie(:,1), exga_1_grie(:,2))
plot(exga_2_grie(:,1), exga_2_grie(:,2))
legend("GA", "CCGA", "EXGA_1", "EXGA_2")
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
plot(exga_1_ackl(:,1), exga_1_ackl(:,2))
plot(exga_2_ackl(:,1), exga_2_ackl(:,2))
legend("GA", "CCGA", "EXGA_1", "EXGA_2")
subtitle("Ackley Function")
ylabel("Best Individual")
xlabel("Function Evaluations")
xlim([0 100000])
ylim([0 16])
