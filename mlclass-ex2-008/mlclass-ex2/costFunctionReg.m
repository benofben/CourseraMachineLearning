function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:length(X)
  x=X(i,:);
  hx = sigmoid(x * theta);
  J += -y(i) * log(hx) - (1-y(i)) * log(1-hx);
endfor
J/=m;

t = theta(2:size(theta));
reg = lambda/(2*m) * sum(t.^2);

J+=reg;

for i = 1:length(X);
  x=X(i,:);
  hx = sigmoid(x * theta);
  grad += (hx - y(i)) * x';
endfor
grad/=m;

for j = 2:length(grad)
  grad(j) += lambda/m*theta(j);
endfor

% =============================================================

end
