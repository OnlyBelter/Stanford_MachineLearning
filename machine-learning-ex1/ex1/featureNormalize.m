function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));  % 1x2
sigma = zeros(1, size(X, 2));  % 1x2


% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset(减去平均值),
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation(除以标准差), storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% X_size = size(X);
for iter = 1:length(mu)
  mu(iter) = mean(X(:, iter));  %表示每个feature的均值
  sigma(iter) = std(X(:, iter));  %表示每个feature的标准差
  X_norm(:, iter) = X_norm(:, iter) - mu(iter);  %每一列减去相应特征的均值
  X_norm(:, iter) = X_norm(:, iter) / sigma(iter); %减去均值后，除以标准差
end


% ============================================================

end
