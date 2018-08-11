% Title: rk4sys.m
% Author: Jun Hao Hu, University of California San Diego
% Date: August 08, 2018
% Purpose: MATLAB procedure that performs the Runge-Kutta fourth order numerical method for the purpose of solving systems of ODEs
% Inputs:
% a, the origin point,
% b, the terminal point,
% h, the step size,
% w0, vector of initial/boundary conditions
% f, function handle containing functions be used in the scheme

function [w] = rk4sys()
