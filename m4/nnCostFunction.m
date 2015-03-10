function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
yy=zeros(length(y),num_labels);
for j=1:num_labels
for i=1:length(y)
yy(i,j)= y(i)==j;
end
end
y=yy;
m = size(X, 1);
t1=ones(m,1);
t2=ones(1,m);
Hyp=sigmoid(Theta2*[t2 ; sigmoid(([t1 X]*Theta1')')] ) ;
Hyp=Hyp';
y;
% You need to return the following variables correctly 
J = (1/m)*sum(sum(-y.*log(Hyp)-(1-y).*log(1-Hyp)))+(lambda/(2*m))*(sum(sum(Theta1(:,2:(input_layer_size+1)).^2))+sum(sum(Theta2(:,2:(hidden_layer_size+1)).^2)))
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Del1=zeros(hidden_layer_size,input_layer_size+1);
Del2=zeros(num_labels,hidden_layer_size+1);
for i=1:m
a1=[1 X(i,:)]';
z2=Theta1*a1;
a2=[1; sigmoid(z2)];
a3=sigmoid(Theta2*a2);
del3=a3-y(i,:)';
del2=(Theta2(:,2:end)'*del3).*sigmoidGradient(z2);
Del2=Del2+del3*a2';
Del1=Del1+del2(1:end)*a1';
end
Theta1_grad=Del1/m +(lambda/m)*[zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad=Del2/m +(lambda/m)*[zeros(num_labels,1) Theta2(:,2:end)];
grad=[Theta1_grad(:);Theta2_grad(:)];
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients



end
