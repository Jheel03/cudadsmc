figure
hold on;
boundary = load("boundaryNodes.txt");
pgon = polyshape(boundary(:, 1),boundary(:, 2));
plot(pgon);
xlim([-2e-6, 1.2e-5]);
ylim([-2e-6, 1.2e-5]);
for i = 1:200
    str = i + ".txt";
    plotData = load(str);
    h1 = scatter(plotData(:, 1), plotData(:, 2), '.r');
    title(str);
    pause(0.2);
    delete(h1);
end