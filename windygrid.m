clear, clc, close all

% This is an implementation of on-policy Sarsa that finds the optimal
% route across a 'windy' gridworld from an initial to a goal state. The
% agent is punished with a reward of -1 on each timestep as usual until
% termination.

% initialize world
world = ones(20, 30) * diag(randi([-1 1], 30, 1)); % -1 : south, 1 : north
world = zeros(20, 30);
start = {3 1};
finish = [17 26];
[m, n] = size(world);

% initialization (rowpos, colpos[, rowstep, colstep])
Q = zeros(m, n, 3, 3);
policy = ones(m, n);
alpha = 0.1;

count = 0;

converging = true;
while converging
    % initialize agent
    [row, col] = start{:};
    action = datasample([randi(9) policy(row, col)], 1, 'Weights', [0.1 0.9]);
    [rstep, cstep] = ind2sub([3 3], action);
    
    episode = [row col rstep cstep]';
    ep = [zeros(4, 1) episode];
    
    tracking_goal_state = true;
    while tracking_goal_state
        
        % generate S' and A'
        row = bound(row + world(row, col) + rstep - 2, m);
        col = bound(col + cstep - 2, n);
        action = datasample([randi(9) policy(row, col)], 1, ...
            'Weights', [0.1 0.9]); % epsilon greedy
        [rstep, cstep] = ind2sub([3 3], action);
        
        episode = [row col rstep cstep]';
        ep = [ep(:, 2) episode];
        
        % check to see if goal was reached
        if isequal([row col], finish)
            tracking_goal_state = false;
        end
        
        % policy evaluation
        SA = sub2ind(size(Q), ep(1, 1), ep(2, 1), ep(3, 1), ep(4, 1));
        SAnext = sub2ind(size(Q), row, col, rstep, cstep);
        Q(SA) = Q(SA) + alpha * (-1 + Q(SAnext) - Q(SA));
        
        % policy improvement
        [~, action] = max(Q(ep(1, 1), ep(2, 1), :), [], 3);
        policy(ep(1, 1), ep(2, 1)) = squeeze(action);
    end
    
    count = count + 1;
end