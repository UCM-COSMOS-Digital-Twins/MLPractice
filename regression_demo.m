%% REGRESSION_DEMO
% Compares 3 supervised regression algorithms on the classic MATLAB
% 'carbig' dataset (built-in, no download needed): predicting MPG from
% Weight and Horsepower.
%   1) Linear Regression
%   2) Ridge Regression and Lasso Regression
%   3) Support Vector Regression (SVR)
%
% Requires: Statistics and Machine Learning Toolbox

clear; clc; close all;

%% Load data
load carbig                       % loads Weight, Horsepower, MPG, etc.
keep = ~isnan(Weight) & ~isnan(Horsepower) & ~isnan(MPG);
X = [Weight(keep), Horsepower(keep)];
y = MPG(keep);

rng(1);
cv = cvpartition(numel(y), 'HoldOut', 0.3);
Xtrain = X(training(cv), :);  ytrain = y(training(cv));
Xtest  = X(test(cv), :);      ytest  = y(test(cv));

% Standardize features (important for Ridge/Lasso/SVR)
mu = mean(Xtrain); sigma = std(Xtrain);
XtrainS = (Xtrain - mu) ./ sigma;
XtestS  = (Xtest - mu) ./ sigma;

%% 1) Linear Regression (ordinary least squares)
mdlLin = fitlm(Xtrain, ytrain);
predLin = predict(mdlLin, Xtest);
rmseLin = sqrt(mean((predLin - ytest).^2));
fprintf('Linear Regression      RMSE = %.2f MPG\n', rmseLin);

%% 2) Ridge and Lasso Regression
% NOTE ON LAMBDA SELECTION: picking lambda by whichever value happens to
% minimize error is easy to get wrong two ways: (a) minimizing error on
% the TEST set leaks test information into model selection, and (b) even
% done properly via cross-validation, the *strict minimum-error* lambda
% on a low-noise, informative feature set like this one often lands so
% close to zero that Ridge/Lasso end up numerically indistinguishable
% from plain OLS. The standard fix practitioners use is the "1-SE rule"
% (popularized by glmnet): pick the MOST regularized lambda whose
% cross-validated error is still within one standard error of the best
% score. That gives a simpler, visibly-shrunk model without sacrificing
% real predictive accuracy -- which is exactly what we want for a class
% demo where the point is to SEE shrinkage happen.
lambdas = logspace(-3, 2, 50);
nFolds = 5;

% ---- Ridge, with a hand-rolled k-fold CV + 1-SE rule ----
% (ridge() itself has no built-in CV, unlike lasso() below)
rng(2);
cvFolds = cvpartition(numel(ytrain), 'KFold', nFolds);
foldMSE = zeros(nFolds, numel(lambdas));
for f = 1:nFolds
    trIdx = training(cvFolds, f);  teIdx = test(cvFolds, f);
    bFold = ridge(ytrain(trIdx), XtrainS(trIdx,:), lambdas, 0);
    predFold = [ones(sum(teIdx),1), XtrainS(teIdx,:)] * bFold;
    foldMSE(f,:) = mean((predFold - ytrain(teIdx)).^2, 1);
end
meanMSE = mean(foldMSE, 1);
seMSE = std(foldMSE, 0, 1) / sqrt(nFolds);
[minMSE, iMin] = min(meanMSE);
withinOneSE = find(meanMSE <= minMSE + seMSE(iMin));
iRidge1SE = withinOneSE(end);      % largest lambda (most regularization) still within 1 SE

bRidgeAll = ridge(ytrain, XtrainS, lambdas, 0);   % refit on full training set, all lambdas
bRidge = bRidgeAll(:, iRidge1SE);
predsRidge = [ones(size(XtestS,1),1), XtestS] * bRidge;
rmseRidge = sqrt(mean((predsRidge - ytest).^2));
fprintf('Ridge Regression        RMSE = %.2f MPG (1-SE lambda=%.4f)\n', rmseRidge, lambdas(iRidge1SE));
fprintf('   -> Ridge coefficients (Weight, Horsepower): [%.3f, %.3f]\n', bRidge(2:3));

% ---- Lasso: L1 penalty, can shrink coefficients all the way to zero.
% lasso()'s built-in CV conveniently reports fitInfo.Index1SE directly.
[bLasso, fitInfo] = lasso(XtrainS, ytrain, 'Lambda', lambdas, 'CV', nFolds);
bestLassoIdx = fitInfo.Index1SE;
predLasso = XtestS * bLasso(:, bestLassoIdx) + fitInfo.Intercept(bestLassoIdx);
rmseLasso = sqrt(mean((predLasso - ytest).^2));
fprintf('Lasso Regression        RMSE = %.2f MPG (1-SE lambda=%.4f)\n', rmseLasso, fitInfo.Lambda(bestLassoIdx));
fprintf('   -> Lasso coefficients (Weight, Horsepower): [%.3f, %.3f]\n', bLasso(:,bestLassoIdx));
fprintf('   -> (compare to OLS coefficients below -- Ridge/Lasso should now visibly shrink toward zero)\n');
disp(mdlLin.Coefficients(2:3, 'Estimate'));

%% 3) Support Vector Regression
mdlSVR = fitrsvm(XtrainS, ytrain, 'KernelFunction', 'gaussian', 'Standardize', false);
predSVR = predict(mdlSVR, XtestS);
rmseSVR = sqrt(mean((predSVR - ytest).^2));
fprintf('Support Vector Regr.    RMSE = %.2f MPG\n', rmseSVR);

%% ---- Visualize: MPG vs Weight, all models overlaid ----
% IMPORTANT: Weight and Horsepower are correlated (~0.86) in real cars --
% light cars almost never have high horsepower. Earlier we sliced the
% curve by holding Horsepower fixed at its GLOBAL mean while sweeping
% Weight across its full range. That creates unrealistic (Weight, HP)
% combinations at the low end (a light car with average horsepower barely
% exists in the data), which sends models -- especially SVR, which has no
% global linear trend to fall back on outside the training distribution
% -- off in strange directions.
%
% Fix: instead of a fixed Horsepower, use the horsepower a car of that
% weight would TYPICALLY have (a simple linear fit of HP given Weight),
% so every plotted point stays close to the real data manifold.
hpGivenWeight = fitlm(X(:,1), X(:,2));
wRange = linspace(min(X(:,1)), max(X(:,1)), 200)';
hpAtWeight = predict(hpGivenWeight, wRange);
gridRaw = [wRange, hpAtWeight];
gridS = (gridRaw - mu) ./ sigma;

yLinCurve   = predict(mdlLin, gridRaw);
yRidgeCurve = [ones(size(gridS,1),1), gridS] * bRidge;
yLassoCurve = gridS * bLasso(:, bestLassoIdx) + fitInfo.Intercept(bestLassoIdx);
ySVRCurve   = predict(mdlSVR, gridS);

figure('Name','Regression Comparison','Position',[100 100 900 600]);
scatter(X(:,1), y, 15, [0.6 0.6 0.6], 'filled'); hold on;
plot(wRange, yLinCurve,   'b-',  'LineWidth', 2);
plot(wRange, yRidgeCurve, 'r--', 'LineWidth', 2);
plot(wRange, yLassoCurve, 'g-.', 'LineWidth', 2);
plot(wRange, ySVRCurve,   'm:',  'LineWidth', 2.5);
xlabel('Weight (lbs)'); ylabel('MPG');
title({'MPG vs. Weight', 'Horsepower held at the value typical for that weight (not a fixed global average)'});
legend('Data', 'Linear', 'Ridge', 'Lasso', 'SVR', 'Location', 'northeast');
grid on;

%% ---- Discussion prompts for class ----
% 1) The two heaviest points in the dataset are >4500 lbs. If I ask you
%    to predict MPG for a 6000-lb car, which curve would you trust and
%    which would you NOT trust? (extrapolation risk)
% 2) Look at the Ridge/Lasso coefficient print-outs vs. the OLS
%    coefficients: do Ridge/Lasso pull the coefficients noticeably
%    toward zero? Did Lasso zero out either feature completely -- what
%    would that mean physically?
% 3) SVR ignores small errors (points "close enough" don't get penalized)
%    -- where does its curve diverge most from Linear Regression?
% 4) We deliberately plotted Horsepower at a "typical" value for each
%    Weight instead of a fixed constant. Why would holding a correlated
%    feature at an unrealistic value make a model's predictions
%    untrustworthy, even inside the range of the training data?
