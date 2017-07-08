function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ... % 20x20 Input Images of Digits
                                   hidden_layer_size, ... % 25 hidden units
                                   num_labels, ...  % 10 labels, from 1 to 10  
                                   X, y, lambda) 
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));  % 25 * 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  % 10 * 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 计算cost
A1 = [ones(m, 1) X]';  % 401·m
Z2 = Theta1 * A1;  % 25·m
A2 = [ones(1, m); sigmoid(Z2)]; %26·m 
Z3 = Theta2 * A2;  % 10·m
h = sigmoid(Z3)'; % A3'


y_m = (y==[1:num_labels]);  % now, y is a matrix, m·10
Theta1_no_t0 = Theta1(:, 2:end);
Theta2_no_t0 = Theta2(:, 2:end);
J = (1/m)*sum((-y_m.*log(h) - (1-y_m).*log(1-h))(:)) + (lambda/(2*m))*(sum((Theta1_no_t0.*Theta1_no_t0)(:)) + sum((Theta2_no_t0.*Theta2_no_t0)(:)));


% 计算所有参数的偏导数（梯度）
delta_2 = 0;
delta_1 = 0;
for t = 1:m
  a_1 = [1 X(t, :)]';  % 401·1
  z_2 = Theta1 * a_1; % 25·1
  a_2 = [1; sigmoid(z_2)]; % 26·1
  z_3 = Theta2 * a_2; % 10·1
  a_3 = sigmoid(z_3);
  h = a_3;  %本身就是列向量
  e_3 = h - y_m(t, :)';  % 10·1
  e_2 = Theta2_no_t0' * e_3 .* sigmoidGradient(z_2); % 25·1
  delta_2 = delta_2 + e_3 * (a_2)';  % 10·26
  delta_1 = delta_1 + e_2 * (a_1)';  % 25·401
  
Theta1_grad(:,1) = (1/m)*delta_1(:,1);
Theta2_grad(:,1) = (1/m)*delta_2(:,1);
Theta1_grad(:,2:end) = (1/m)*delta_1(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = (1/m)*delta_2(:,2:end) + (lambda/m)*Theta2(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
