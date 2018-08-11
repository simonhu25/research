% Title: rk4.m
% Author: Jun Hao Hu, University of California San Diego
% Date: August 08, 2018
% Purpose: MATLAB procedure that performs the Runge-Kutta fourth order scheme, for the purpose of solving ODEs
% Inputs:
% a, the origin point,
% b, the terminal point,
% h, the step size,
% w0, the initial condition,
% f, a function handle containing

function [w] = rk4(a,b,h,w0,f)

w = zeros(1, abs(b-a)/h); % vector that stores the output values
t = a:h:b+h; % the uniform mesh, with each grid point spaced apart by h units
w(1) = w0; % encode the initial condition
f = f{1}; % define the function f

% implement the RK4 scheme
for i = 1:size(w,2)
    k1 = h*f(t(i),w(i));
    k2 = h*f(t(i)+h/2,w(i)+k1/2);
    k3 = h*f(t(i)+h/2,w(i)+k2/2);
    k4 = h*f(t(i)+h,w(i)+k3);
    w(i+1) = w(i)+(1/6)*(k1+2*k2+2*k3+k4);
end
