filePathPrefix = "C:/Users/csyzz/Projects/TimeIntegrator/build/output/Pogo_Stick/";
integratorTypes = ["Implicit Euler", "Newmark", "TRBDF2"', "BDF2"];
fileName = "/simulation_status.txt";
[~, cols] = size(integratorTypes);
close all;

figure;
for i = 1 : cols
    hold on;
    datafile = filePathPrefix + integratorTypes(i) + fileName;
    A = readmatrix(datafile);
    plots(i) = plot(A(:, 1), A(:, 2));
end
legend(plots,  integratorTypes);
xlabel('t/s');
ylabel('spring potential');
hold off;

figure;
for i = 1 : cols
    hold on;
    datafile = filePathPrefix + integratorTypes(i) + fileName;
    A = readmatrix(datafile);
    plots(i) = plot(A(:, 1), A(:, 3));
end
legend(plots,  integratorTypes);
xlabel('t/s');
ylabel('gravity potential');
hold off;

figure;
for i = 1 : cols
    hold on;
    datafile = filePathPrefix + integratorTypes(i) + fileName;
    A = readmatrix(datafile);
    plots(i) = plot(A(:, 1), A(:, 4));
end
legend(plots,  integratorTypes);
xlabel('t/s');
ylabel('IPC barier');
hold off;

figure;
for i = 1 : cols
    hold on;
    datafile = filePathPrefix + integratorTypes(i) + fileName;
    A = readmatrix(datafile);
    plots(i) = plot(A(:, 1), A(:, 5));
end
legend(plots,  integratorTypes);
xlabel('t/s');
ylabel('kinetic energy');
hold off;

figure;
for i = 1 : cols
    hold on;
    datafile = filePathPrefix + integratorTypes(i) + fileName;
    A = readmatrix(datafile);
    plots(i) = plot(A(:, 1), A(:, 6));
end
legend(plots,  integratorTypes);
xlabel('t/s');
ylabel('center of mass');
hold off;