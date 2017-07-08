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
%               of the feature and subtract it from the dataset(��ȥƽ��ֵ),
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation(���Ա�׼��), storing
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
  mu(iter) = mean(X(:, iter));  %��ʾÿ��feature�ľ�ֵ
  sigma(iter) = std(X(:, iter));  %��ʾÿ��feature�ı�׼��
  X_norm(:, iter) = X_norm(:, iter) - mu(iter);  %ÿһ�м�ȥ��Ӧ�����ľ�ֵ
  X_norm(:, iter) = X_norm(:, iter) / sigma(iter); %��ȥ��ֵ�󣬳��Ա�׼��
end


% ============================================================

end
