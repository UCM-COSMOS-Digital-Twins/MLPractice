%% REINFORCEMENT_DEMO
% Part 1: Q-Learning from scratch on a simple 5x5 gridworld (no toolbox
%         needed -- great for showing students the raw update rule).
% Part 2: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) on
%         MATLAB's built-in Cart-Pole environment.
%
% Part 1 requires: base MATLAB only
% Part 2 requires: Reinforcement Learning Toolbox, Deep Learning Toolbox
%         (Part 2 is commented out to run by default, since a live
%          training run can take several minutes -- see notes below.)

clear; clc; close all;

%% ================= PART 1: Q-LEARNING FROM SCRATCH =================
% Gridworld layout (5x5). Agent starts at (1,1), goal at (5,5).
% Reward: -1 per step (encourages speed), +10 for reaching the goal.
gridSize = 5;
goal = [5 5];
nActions = 4;                 % 1=up, 2=down, 3=left, 4=right
actionDelta = [-1 0; 1 0; 0 -1; 0 1];

Q = zeros(gridSize, gridSize, nActions);   % Q-table: state x action

alpha = 0.5;    % learning rate
gamma = 0.9;    % discount factor
epsilon = 0.2;  % exploration rate
nEpisodes = 500;
episodeLengths = zeros(nEpisodes,1);

rng(1);
for ep = 1:nEpisodes
    state = [1 1];
    steps = 0;
    while ~isequal(state, goal) && steps < 100
        steps = steps + 1;

        % epsilon-greedy action selection
        if rand < epsilon
            a = randi(nActions);
        else
            qVals = squeeze(Q(state(1), state(2), :));
            [~, a] = max(qVals);
        end

        % take action, clip to grid boundaries
        nextState = state + actionDelta(a, :);
        nextState = max(min(nextState, gridSize), 1);

        % reward
        if isequal(nextState, goal)
            r = 10;
        else
            r = -1;
        end

        % Q-learning update rule (the heart of the algorithm)
        bestNextQ = max(squeeze(Q(nextState(1), nextState(2), :)));
        Q(state(1), state(2), a) = Q(state(1), state(2), a) + ...
            alpha * (r + gamma*bestNextQ - Q(state(1), state(2), a));

        state = nextState;
    end
    episodeLengths(ep) = steps;
end

figure('Name','Q-Learning','Position',[100 100 1100 450]);
subplot(1,2,1);
plot(movmean(episodeLengths, 20), 'LineWidth', 2);
xlabel('Episode'); ylabel('Steps to reach goal (20-episode moving avg)');
title('Q-Learning: Getting Faster With Experience');
grid on;

% Visualize the learned policy (best action per cell) as arrows.
% state = [row, col]; gx/gy from meshgrid give gx=col, gy=row directly
% with NO manual flipping -- the previous version flipped the goal
% marker's row (gridSize-goal(1)+1) but left the quiver arrows unflipped,
% so the star landed at (5,1) instead of (5,5) where the arrows actually
% converge. Fix: keep row/col consistent everywhere, and use `axis ij`
% (image-style: row 1 at top, increasing downward) to get a natural
% grid-world layout instead of a manual per-element flip.
[~, bestAction] = max(Q, [], 3);
subplot(1,2,2);
[gx, gy] = meshgrid(1:gridSize, 1:gridSize);   % gx = col, gy = row
arrowU = zeros(size(bestAction)); arrowV = zeros(size(bestAction));
for i = 1:gridSize
    for j = 1:gridSize
        d = actionDelta(bestAction(i,j), :);   % d = [drow, dcol]
        arrowU(i,j) = d(2);   % column delta -> horizontal component
        arrowV(i,j) = d(1);   % row delta -> vertical component
    end
end
quiver(gx, gy, arrowU, arrowV, 0.5, 'LineWidth', 1.5);
hold on; plot(goal(2), goal(1), 'rp', 'MarkerSize', 20, 'MarkerFaceColor','r');
axis equal tight; axis ij;   % row 1 at top, matching typical grid-world layout
title('Learned Policy: Arrow = Best Action Per Cell');
xlabel('column'); ylabel('row');

%% ---- Discussion prompts for Part 1 ----
% 1) Every arrow should roughly point toward the red goal star. Do any
%    look wrong? Why might a rarely-visited cell have a bad policy?
% 2) What happens to the learning curve if you set epsilon = 0 (never
%    explore)? Try it.
% 3) The reward is -1 per step, +10 at the goal. What behavior would
%    change if the goal reward were only +1?

%% ================= PART 2: DQN AND PPO (Cart-Pole) =================
% This section trains on MATLAB's built-in Cart-Pole balancing task using
% Reinforcement Learning Toolbox. Training can take several minutes even
% on this small problem, so it's left commented out -- uncomment to run
% live in class, or run beforehand and show the result video/plot.
%
% The key teaching point: DQN and Q-learning use the SAME idea (learn a
% Q-value for state-action pairs), but DQN replaces the table with a
% neural network so it can handle the CONTINUOUS cart-pole state (cart
% position/velocity, pole angle/angular velocity) that a table can't.
% PPO instead learns a policy network directly, with clipped updates
% for stability -- the current industry-standard RL algorithm.

% --- Environment setup (shared by DQN and PPO) ---
env = rlPredefinedEnv("CartPole-Discrete");   % for DQN (discrete actions)
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
%
% --- DQN agent ---
% WHY THE REWARD CURVE ISN'T MONOTONIC: DQN keeps exploring (epsilon-
% greedy) and keeps updating its network for every episode you train, so
% there is no guarantee the LAST episode is the BEST episode -- it can
% find a great policy at episode 150 and then wander away from it by
% episode 300 (a mix of continued random exploration and the well-known
% instability of bootstrapped Q-learning with function approximation).
% Two independent fixes:
%   (1) Tune the agent to wander less once it finds a good policy.
%   (2) Checkpoint agents during training and keep the best one, instead
%       of trusting whichever agent happens to exist when training stops.
qNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(24)
    reluLayer
    fullyConnectedLayer(24)
    reluLayer
    fullyConnectedLayer(numel(actInfo.Elements))];
critic = rlVectorQValueFunction(qNetwork, obsInfo, actInfo);

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN', true, ...              % reduces DQN's well-known overestimation bias
    'TargetSmoothFactor', 1e-3, ...        % slow, smooth target-network updates instead of hard periodic copies -> less oscillation
    'ExperienceBufferLength', 1e5, ...     % bigger replay buffer = less correlated, less "forgetting" of earlier good experience
    'MiniBatchSize', 128, ...
    'DiscountFactor', 0.99);
agentOpts.EpsilonGreedyExploration.Epsilon = 1;         % start fully exploratory
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.01;   % but decay to nearly greedy...
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3; % ...so late episodes reflect the LEARNED policy, not residual randomness
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1; % clip gradients -> fewer destructive updates

agentDQN = rlDQNAgent(critic, agentOpts);
%
% --- Checkpoint every agent that clears a reward threshold, to disk ---
% SaveAgentCriteria/SaveAgentValue write a .mat file for every episode
% whose reward crosses the threshold, into SaveAgentDirectory. After
% training, you can load and evaluate EVERY checkpoint and keep the best
% one, rather than being stuck with whatever the final episode produced.
trainOpts = rlTrainingOptions('MaxEpisodes', 300, ...
    'StopTrainingCriteria', 'AverageReward', 'StopTrainingValue', 480, ...
    'ScoreAveragingWindowLength', 20, ...      % smooths the stopping criterion over 20 episodes, not 1 lucky episode
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 400, ...
    'SaveAgentDirectory', 'savedAgentsDQN');
trainingStatsDQN = train(agentDQN, env, trainOpts);
%
% --- Recover the best checkpoint (not just the final agent) ---
checkpointFiles = dir(fullfile('savedAgentsDQN', 'Agent*.mat'));
bestReward = -inf; bestAgentFile = '';
simOptsEval = rlSimulationOptions('MaxSteps', 500);
for k = 1:numel(checkpointFiles)
    s = load(fullfile(checkpointFiles(k).folder, checkpointFiles(k).name));
    % evaluate over a few episodes since performance can be noisy
    rewards = zeros(1,3);
    for trial = 1:3
        exp = sim(env, s.saved_agent, simOptsEval);
        rewards(trial) = sum(exp.Reward);
    end
    avgReward = mean(rewards);
    if avgReward > bestReward
        bestReward = avgReward;
        bestAgentFile = checkpointFiles(k).name;
        bestAgentDQN = s.saved_agent;
    end
end
fprintf('Best checkpoint: %s (avg reward over 3 eval runs = %.1f)\n', bestAgentFile, bestReward);
% % Use bestAgentDQN (not agentDQN) for the final sim/plot below.
%
% --- PPO agent (uses continuous CartPole environment) ---
envC = rlPredefinedEnv("CartPole-Continuous");
agentPPO = rlPPOAgent(getObservationInfo(envC), getActionInfo(envC));
trainingStatsPPO = train(agentPPO, envC, trainOpts);
% % Note: PPO is inherently more stable episode-to-episode than DQN,
% % since it clips how far each policy update is allowed to move --
% % that's the "P" in PPO. It's still good practice to checkpoint it too.
%
% --- Compare learning curves ---
figure; hold on;
plot(trainingStatsDQN.EpisodeReward, 'DisplayName','DQN');
plot(trainingStatsPPO.EpisodeReward, 'DisplayName','PPO');
legend; xlabel('Episode'); ylabel('Reward'); title('DQN vs PPO on Cart-Pole');
%
%% --- Watch the TRAINED agents actually balance the pole ---
% This is usually the best "wow" moment of the whole lecture -- an
% animated window pops up showing the cart-pole balancing live. Call
% plot(env) BEFORE sim() to open the visualizer; sim() will then animate
% into that window step by step as the trained agent acts.
simOpts = rlSimulationOptions('MaxSteps', 500);

plot(env);                                   % open the DQN visualizer window
agentToShow = agentDQN;
if exist('bestAgentDQN', 'var')
    agentToShow = bestAgentDQN;   % prefer the best checkpoint if one was recovered above
end
experienceDQN = sim(env, agentToShow, simOpts);
totalRewardDQN = sum(experienceDQN.Reward);
fprintf('Trained DQN agent: total reward = %.1f (max possible = 500)\n', totalRewardDQN);
%
plot(envC);                                  % open the PPO visualizer window
experiencePPO = sim(envC, agentPPO, simOpts);
totalRewardPPO = sum(experiencePPO.Reward);
fprintf('Trained PPO agent: total reward = %.1f (max possible = 500)\n', totalRewardPPO);

% --- Optional: save agents so you don't have to retrain before class ---
% save('cartpole_agents.mat', 'agentDQN', 'agentPPO');
% % ... and later, to reload without retraining:
% % load('cartpole_agents.mat', 'agentDQN', 'agentPPO');

%% ---- Discussion prompts for Part 2 ----
% 1) Why can't we use a Q-TABLE (like Part 1) for Cart-Pole? (Hint: how
%    many possible states are there if position/angle are continuous?)
% 2) DQN learns VALUES (how good is each action?). PPO learns a POLICY
%    directly (what action should I take?). What's the practical
%    difference in how you'd use each one at decision time?
% 3) PPO is what's used to fine-tune large language models with human
%    feedback (RLHF) -- the "reward" there is a human preference model
%    instead of a physics simulator. Same algorithm, wildly different
%    application.
% 4) Plot trainingStatsDQN.EpisodeReward and find its peak -- is it at
%    the last episode, or somewhere in the middle? What does that tell
%    you about why "just use the final agent" is a risky default for
%    DQN, and why checkpointing + picking the best saved agent is
%    standard practice rather than an edge case?
