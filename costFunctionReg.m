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
L = eye(size(theta, 1));
L(1,1) = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J_fit = -1/m*((y'*log(sigmoid(X*theta)))+((1-y)'*log(1-sigmoid(X*theta))));
J_reg = lambda/(2*m)*((L*theta)'*(L*theta));
J = J_fit + J_reg;
cost = J;

for i = 1:size(X,2),
    grad(i) = (1/m*sum((sigmoid(X*theta)-y).*X(:,i)))+ (lambda*L(i,i)*theta(i)/m);
end;


% =============================================================

end
