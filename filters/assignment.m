%% Exercises Module VIII: Advanced supervised gradient learning methods - Part II: filter algorithms - Pawel Chilinski
% The implementation of the Paralel Kalman Filter and Recursive Least
% Squares. The algorithms are compared on simple XOR problem in the same
% way as in orginal papers where methods were described.

%% Training XOR data
zero=-0.95;one=0.95;
data=[zero zero zero;one one zero;zero one one;one zero one];

%%
% Split it into input and output:
X=data(:,1:2);
Y=data(:,3);

%% 
% number of epochs
epochs=100;

%% Paramlel kalman filter
pkfNn=ffnn([2;10;1]);

pkfLearner=pkf(pkfNn);

pkfLearner.X = X;
pkfLearner.Y = Y;
pkfLearner.epochs = epochs;
pkfLearner.learn();

plot(pkfLearner.diagnostics.trainRmse);
title('Training RMSE using Parallel Kalman Filter');
xlabel('epoch');
ylabel('RMSE');

%% Recursive least squares
rlsNn=ffnn([2;10;1]);

rlsLearner=rls(rlsNn);

rlsLearner.X = X;
rlsLearner.Y = Y;
rlsLearner.epochs = epochs;
rlsLearner.learn();

figure;
plot(rlsLearner.diagnostics.trainRmse);
title('Training RMSE using Recursive Least Squares');
xlabel('epoch');
ylabel('RMSE');

%% Conclusions
% Recursive Least Squares converges quicker than Parallel Kalman Filter.