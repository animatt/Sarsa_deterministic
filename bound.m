function bounded_output = bound(expression, limit)
if expression < 1
    bounded_output = 1;
elseif expression > limit
    bounded_output = limit;
else
    bounded_output = expression;
end
end