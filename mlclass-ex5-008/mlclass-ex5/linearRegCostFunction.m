function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

for i = 1:size(X,1)
  x=X(i,:);
  hx = x * theta;
  J += (hx - y(i))^2;
endfor
J/=2*m;

t = theta(2:size(theta));
reg = lambda/(2*m) * sum(t.^2);

J+=reg;

for i = 1:size(X,1);
  x=X(i,:);
  hx = x * theta;
  grad += (hx - y(i)) * x';
endfor
grad/=m;

for j = 2:length(grad)
  grad(j) += lambda/m*theta(j);
endfor


% =========================================================================

grad = grad(:);

end
