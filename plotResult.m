function plotResult(outputSize, rSquare_train,rSquare_test)
% plots the R square score Between Y0 and network output per neuron

figure('units','normalized', 'position', [0.2,0.2, 0.6, 0.6]);
plot(1 : outputSize, rSquare_train(end : -1 : 1))
hold on
plot(1 : outputSize, rSquare_test(end : -1 : 1))
title('R Square Score Of Output Neurons With Matching Y0')
xlabel('Neuron')
ylabel('R Square')
legend({'Train', 'Test'})
end