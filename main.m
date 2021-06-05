%% Exercise 3- Echo State Network (ESN)
% yair lahad
clear; clc;

%% Parameters
% Network includes the input layer W_in, reservoir layer K
% output layer W_out and reservoir state r
actFunc = @ActFuncs.Tanh;
alpha=1;           % leaking rate
betta=1e-8;        % regularization coefficient
regInput=0.01;     % regularization for W_in
regHidden=0.98;    % regularization for K
Sparse=0.3;        % percentage of zero weights of K
% Network structure
inputSize = 1;
hiddenSize = 500;
outputSize = 100;
samplesSize=10000;
trialsSize=samplesSize-outputSize;

%% Build the ESN
%input layer
X= rand(1,samplesSize);
W_in= rand(hiddenSize,inputSize)*regInput;

%reservoir layer
K= randn(hiddenSize) .* randi([0,1],hiddenSize);
rhow= max(real(eig(K))); % biggest eigenvalue
K= (K./rhow)* regHidden; % update K
r= zeros(hiddenSize,1);  % reservoir state
% Matrice for each time step, saves the reservoier state (with bias & input neurons)
r_mat=zeros(trialsSize,hiddenSize+2);

% output layer
Y0=zeros(outputSize,trialsSize);   % teachers for predictions
rSquare_train=zeros(outputSize,1);
rSquare_test=zeros(outputSize,1);

%% Learning Process of the ESN
% feed forward
for i = 1:length(X)
    r = (1-alpha)*r + alpha*actFunc(W_in*X(i)+K*r);  % Dynamic equation of recurrent network
    if i > outputSize
        currInd=i-outputSize;
        Y0(:,currInd)=X(currInd:i-1)';   % update Y0
        r_mat(currInd,:) = [r;X(i);1];   % update reservoir state + input and bias
    end
end
% learning of W_out using linear regression
C = (1/trialsSize) * (r_mat)' * (r_mat);
u = (1/trialsSize)*Y0*r_mat;
% ridge regression
W_out = (C + betta * eye(size(C, 1))) \ u';
Y = (r_mat*W_out)'; % output of reservoir (Prediction)

%% Computational part
% R square Score per neuron
for neuron = 1 : outputSize
    rSquare_train(neuron) = corr(Y(neuron, :)', Y0(neuron, :)') .^ 2;
end
memoryCapacity=sum(rSquare_train);
disp(['Training Memory Capacity is : ' num2str(memoryCapacity)])

%% ESN Testing
X_test = randn(1,samplesSize);
Y0_test = zeros(outputSize,trialsSize);
r_mat_test = zeros(trialsSize,hiddenSize+2);
% feed forword
for i = 1:length(X)
    r = (1-alpha)*r + alpha*actFunc(W_in*X(i)+K*r);
    if i > outputSize
        currInd=i-outputSize;
        Y0_test(:,currInd)=X(currInd:i-1)';   % update Y0
        r_mat_test(currInd,:) = [r;X(i);1];   % update reservoir state + input and bias
    end
end
Y_test = (r_mat_test*W_out)';
% R square Score per neuron
for neuron = 1 : outputSize
    rSquare_test(neuron) = corr(Y_test(neuron, :)', Y0_test(neuron, :)') .^ 2;
end
memoryCapacity_test=sum(rSquare_test);
disp(['Test Memory Capacity is : ' num2str(memoryCapacity_test)])

%% Plot Result
plotResult(outputSize, rSquare_train,rSquare_test);
