load ..\ga\ga_rast.txt
load ..\ga\ga_schw.txt
load ..\ga\ga_ackl.txt
load ..\ga\ga_grie.txt

ga_plot = figure();

% Rastrigin Function
subplot(2, 2, 1)
plot(ga_rast(:,1), ga_rast(:,2))
subtitle("Rastrigin Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Schwefel Function
subplot(2, 2, 2)
plot(ga_schw(:,1), ga_schw(:,2))
subtitle("Schwefel Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Griewangk Function
subplot(2, 2, 3)
plot(ga_grie(:,1), ga_grie(:,2))
subtitle("Griewangk Function")
ylabel("Best Individual")
xlabel("Function Evaluations")

% Ackley Function
subplot(2, 2, 4)
plot(ga_ackl(:,1), ga_ackl(:,2))
subtitle("Ackley Function")
ylabel("Best Individual")
xlabel("Function Evaluations")