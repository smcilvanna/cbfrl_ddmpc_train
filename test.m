
%DQN2


% Clear previous variables and settings
clear; clc;

% Define context space and action space
numContexts = 10;  % Number of contexts (e.g., age groups)
numActions = 5;    % Number of actions (e.g., articles)

% Neural network parameters
inputSize = numContexts;
outputSize = numActions;
hiddenLayerSize = 24;  % Size of the hidden layer

% Create the Q-network
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(hiddenLayerSize)
    reluLayer
    fullyConnectedLayer(hiddenLayerSize)
    reluLayer
    fullyConnectedLayer(outputSize)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ... % Increase the number of epochs
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-3, ... % Adjust learning rate
    'Verbose',false);

% Define the Gaussian reward function
mu_c = 5; % Mean of context
mu_a = 3; % Mean of action
sigma_c = 2; % Standard deviation of context
sigma_a = 1; % Standard deviation of action
rewardFunction = @(context, action) exp(-((context - mu_c).^2 / (2 * sigma_c^2) + (action - mu_a).^2 / (2 * sigma_a^2)));

% Training parameters
numEpisodes = 10000;
epsilon = 1.0; % Initial exploration rate
epsilonDecay = 0.995; % Decay rate for epsilon
minEpsilon = 0.1; % Minimum value for epsilon
gamma = 0.99; % Discount factor
experienceReplaySize = 10000; % Size of the experience replay buffer
batchSize = 64; % Batch size for training

% Initialize experience replay buffer
experienceReplay = {};

% Initialize Q-network
QNetwork = trainNetwork(zeros(batchSize, inputSize), zeros(batchSize, outputSize), layers, options);

% Training loop
for episode = 1:numEpisodes
    % Initialize context (state)
    context = randi(numContexts);
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    
    % Epsilon-greedy action selection
    if rand < epsilon
        action = randi(numActions);
    else
        qValues = predict(QNetwork, contextVector);
        [~, action] = max(qValues);
    end
    
    % Calculate reward based on the Gaussian function
    reward = rewardFunction(context, action);
    % Map to binary reward (e.g., click if reward > threshold)
    binaryReward = reward > 0.5;
    
    % Store experience in the replay buffer
    experience = {contextVector, action, binaryReward};
    if length(experienceReplay) < experienceReplaySize
        experienceReplay{end+1} = experience;
    else
        experienceReplay{randi(experienceReplaySize)} = experience;
    end
    
    % Sample a random minibatch from the experience replay buffer
    if length(experienceReplay) >= batchSize
        minibatch = datasample(experienceReplay, batchSize);
        contexts = cell2mat(cellfun(@(x) x{1}, minibatch, 'UniformOutput', false));
        actions = cell2mat(cellfun(@(x) x{2}, minibatch, 'UniformOutput', false));
        rewards = cell2mat(cellfun(@(x) x{3}, minibatch, 'UniformOutput', false));
        
        % Reshape contexts to be batchSize x numContexts
        contexts = reshape(contexts, batchSize, numContexts);
        
        % Get the current Q-values and target Q-values
        qValues = predict(QNetwork, contexts);
        targetQValues = qValues;
        for i = 1:batchSize
            targetQValues(i, actions(i)) = rewards(i);
        end
        
        % Train the Q-network
        QNetwork = trainNetwork(contexts, targetQValues, layers, options);
    end
    
    % Decay epsilon
    epsilon = max(minEpsilon, epsilon * epsilonDecay);
end

% Display the learned Q-values
disp('Learned Q-values:');
for context = 1:numContexts
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    qValues = predict(QNetwork, contextVector);
    disp(['Context ', num2str(context), ': ', num2str(qValues)]);
end

% Visualization
figure;

% Subplot 1: Ground Truth Reward Function
subplot(1, 2, 1);
[X, Y] = meshgrid(1:numContexts, 1:numActions);
Z = rewardFunction(X, Y);
surf(X, Y, Z);
title('Ground Truth Reward Function');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Reward');
colorbar;

% Subplot 2: Learned Q-values
subplot(1, 2, 2);
Q_values = zeros(numContexts, numActions);
for context = 1:numContexts
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    Q_values(context, :) = predict(QNetwork, contextVector);
end
surf(1:numContexts, 1:numActions, Q_values');
title('Learned Q-values');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Q-value');
colorbar;

% Use Q-values for prediction (action selection)
% Example prediction for a new user in age group 7
newContext = 7;
contextVector = zeros(1, numContexts); % Create one-hot encoded context
contextVector(newContext) = 1;
qValues = predict(QNetwork, contextVector);
[~, bestAction] = max(qValues);
disp(['For age group ', num2str(newContext), ', show article ', num2str(bestAction)]);




%% DQN1

% Clear previous variables and settings
clear; clc;

% Define context space and action space
numContexts = 10;  % Number of contexts (e.g., age groups)
numActions = 5;    % Number of actions (e.g., articles)

% Neural network parameters
inputSize = numContexts;
outputSize = numActions;
hiddenLayerSize = 24;  % Size of the hidden layer

% Create the Q-network
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(hiddenLayerSize)
    reluLayer
    fullyConnectedLayer(hiddenLayerSize)
    reluLayer
    fullyConnectedLayer(outputSize)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',1, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);

% Define the Gaussian reward function
mu_c = 5; % Mean of context
mu_a = 3; % Mean of action
sigma_c = 2; % Standard deviation of context
sigma_a = 1; % Standard deviation of action
rewardFunction = @(context, action) exp(-((context - mu_c).^2 / (2 * sigma_c^2) + (action - mu_a).^2 / (2 * sigma_a^2)));

% Training parameters
numEpisodes = 10000;
epsilon = 1.0; % Initial exploration rate
epsilonDecay = 0.995; % Decay rate for epsilon
minEpsilon = 0.1; % Minimum value for epsilon
gamma = 0.99; % Discount factor
experienceReplaySize = 10000; % Size of the experience replay buffer
batchSize = 64; % Batch size for training

% Initialize experience replay buffer
experienceReplay = {};

% Initialize Q-network
QNetwork = trainNetwork(zeros(batchSize, inputSize), zeros(batchSize, outputSize), layers, options);

% Training loop
for episode = 1:numEpisodes
    % Initialize context (state)
    context = randi(numContexts);
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    
    % Epsilon-greedy action selection
    if rand < epsilon
        action = randi(numActions);
    else
        qValues = predict(QNetwork, contextVector);
        [~, action] = max(qValues);
    end
    
    % Calculate reward based on the Gaussian function
    reward = rewardFunction(context, action);
    % Map to binary reward (e.g., click if reward > threshold)
    binaryReward = reward > 0.5;
    
    % Store experience in the replay buffer
    experience = {contextVector, action, binaryReward};
    if length(experienceReplay) < experienceReplaySize
        experienceReplay{end+1} = experience;
    else
        experienceReplay{randi(experienceReplaySize)} = experience;
    end
    
    % Sample a random minibatch from the experience replay buffer
    if length(experienceReplay) >= batchSize
        minibatch = datasample(experienceReplay, batchSize);
        contexts = cell2mat(cellfun(@(x) x{1}, minibatch, 'UniformOutput', false));
        actions = cell2mat(cellfun(@(x) x{2}, minibatch, 'UniformOutput', false));
        rewards = cell2mat(cellfun(@(x) x{3}, minibatch, 'UniformOutput', false));
        
        % Reshape contexts to be batchSize x numContexts
        contexts = reshape(contexts, batchSize, numContexts);
        
        % Get the current Q-values and target Q-values
        qValues = predict(QNetwork, contexts);
        targetQValues = qValues;
        for i = 1:batchSize
            targetQValues(i, actions(i)) = rewards(i);
        end
        
        % Train the Q-network
        QNetwork = trainNetwork(contexts, targetQValues, layers, options);
    end
    
    % Decay epsilon
    epsilon = max(minEpsilon, epsilon * epsilonDecay);
end

% Display the learned Q-values
disp('Learned Q-values:');
for context = 1:numContexts
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    qValues = predict(QNetwork, contextVector);
    disp(['Context ', num2str(context), ': ', num2str(qValues)]);
end

% Visualization
figure;

% Subplot 1: Ground Truth Reward Function
subplot(1, 2, 1);
[X, Y] = meshgrid(1:numContexts, 1:numActions);
Z = rewardFunction(X, Y);
surf(X, Y, Z);
title('Ground Truth Reward Function');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Reward');
colorbar;

% Subplot 2: Learned Q-values
subplot(1, 2, 2);
Q_values = zeros(numContexts, numActions);
for context = 1:numContexts
    contextVector = zeros(1, numContexts); % Create one-hot encoded context
    contextVector(context) = 1;
    Q_values(context, :) = predict(QNetwork, contextVector);
end
surf(1:numContexts, 1:numActions, Q_values');
title('Learned Q-values');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Q-value');
colorbar;

% Use Q-values for prediction (action selection)
% Example prediction for a new user in age group 7
newContext = 7;
contextVector = zeros(1, numContexts); % Create one-hot encoded context
contextVector(newContext) = 1;
qValues = predict(QNetwork, contextVector);
[~, bestAction] = max(qValues);
disp(['For age group ', num2str(newContext), ', show article ', num2str(bestAction)]);


%% Qtable - modified

% Define context space and action space
contextSpace = 1:10;  % 10 age groups (increased)
actionSpace = 1:0.1:5;   % 5 articles (increased)

% Initialize Q-values (context-action values)
Q = zeros(length(contextSpace), length(actionSpace));

% Define learning parameters
alpha = 0.1;   % Learning rate
epsilon = 0.5; % Initial exploration rate
epsilonDecay = 0.99; % Decay rate for epsilon
minEpsilon = 0.1; % Minimum value for epsilon

% Define the Gaussian reward function
mu_c = 5; % Mean of context
mu_a = 3; % Mean of action
sigma_c = 2; % Standard deviation of context
sigma_a = 1; % Standard deviation of action

rewardFunction = @(context, action) exp(-((context - mu_c).^2 / (2 * sigma_c^2) + (action - mu_a).^2 / (2 * sigma_a^2)));

% Simulate user interaction
numEpisodes = 10000; % Increased number of episodes
for episode = 1:numEpisodes
    % Randomly select a context (user age group)
    context = randi(length(contextSpace));
    
    % Epsilon-greedy action selection
    if rand < epsilon
        action = randi(length(actionSpace));
    else
        [~, action] = max(Q(context, :));
    end
    
    % Calculate reward based on the Gaussian function
    reward = rewardFunction(context, action);
    % Map to binary reward (e.g., click if reward > threshold)
    binaryReward = reward > 0.5;
    
    % Update Q-values
    Q(context, action) = Q(context, action) + alpha * (binaryReward - Q(context, action));
    
    % Decay epsilon
    epsilon = max(minEpsilon, epsilon * epsilonDecay);
end

% Display the learned Q-values
disp('Learned Q-values:');
disp(Q);

% Visualization
figure;

% Subplot 1: Ground Truth Reward Function
subplot(1, 2, 1);
[X, Y] = meshgrid(contextSpace, actionSpace);
Z = rewardFunction(X, Y);
surf(X, Y, Z);
title('Ground Truth Reward Function');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Reward');
colorbar;

% Subplot 2: Learned Q-values
subplot(1, 2, 2);
surf(contextSpace, actionSpace, Q');
title('Learned Q-values');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Q-value');
colorbar;

% Use Q-values for prediction (action selection)
% Example prediction for a new user in age group 7
newContext = 7;
[~, bestAction] = max(Q(newContext, :));
disp(['For age group ', num2str(newContext), ', show article ', num2str(bestAction)]);








%%

%% Q table example
% Define context space and action space
contextSpace = 1:0.01:5;  % 5 age groups
actionSpace = 1:0.01:3;   % 3 articles

% Initialize Q-values (context-action values)
Q = zeros(length(contextSpace), length(actionSpace));

% Define learning parameters
alpha = 0.1;   % Learning rate
epsilon = 0.1; % Exploration rate

% Define the Gaussian reward function
mu_c = 3; % Mean of context
mu_a = 2; % Mean of action
sigma_c = 1; % Standard deviation of context
sigma_a = 0.5; % Standard deviation of action

rewardFunction = @(context, action) exp(-((context - mu_c).^2 / (2 * sigma_c^2) + (action - mu_a).^2 / (2 * sigma_a^2)));

% Simulate user interaction
numEpisodes = 50000;
for episode = 1:numEpisodes
    % Randomly select a context (user age group)
    context = randi(length(contextSpace));
    
    % Epsilon-greedy action selection
    if rand < epsilon
        action = randi(length(actionSpace));
    else
        [~, action] = max(Q(context, :));
    end
    
    % Calculate reward based on the Gaussian function
    reward = rewardFunction(context, action);
    % Map to binary reward (e.g., click if reward > threshold)
    binaryReward = reward > 0.5;
    
    % Update Q-values
    Q(context, action) = Q(context, action) + alpha * (binaryReward - Q(context, action));
end

% Display the learned Q-values
disp('Learned Q-values:');
disp(Q);

% Visualization
figure;

% Subplot 1: Ground Truth Reward Function
subplot(1, 2, 1);
[X, Y] = meshgrid(contextSpace, actionSpace);
Z = rewardFunction(X, Y);
surf(X, Y, Z);
title('Ground Truth Reward Function');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Reward');
colorbar;

% Subplot 2: Learned Q-values
subplot(1, 2, 2);
surf(contextSpace, actionSpace, Q');
title('Learned Q-values');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Q-value');
colorbar;

% Use Q-values for prediction (action selection)
% Example prediction for a new user in age group 3
newContext = 3;
[~, bestAction] = max(Q(newContext, :));
disp(['For age group ', num2str(newContext), ', show article ', num2str(bestAction)]);




















%%






%% OLD
% Define context space and action space
contextSpace = 1:5;  % 5 age groups
actionSpace = 1:3;   % 3 articles

% Initialize Q-values (context-action values)
Q = zeros(length(contextSpace), length(actionSpace));

% Define learning parameters
alpha = 0.1;   % Learning rate
epsilon = 0.1; % Exploration rate

% Define the non-linear reward function
rewardFunction = @(context, action) sin(pi * context .* action);

% Simulate user interaction
numEpisodes = 10000;
for episode = 1:numEpisodes
    % Randomly select a context (user age group)
    context = randi(length(contextSpace));
    
    % Epsilon-greedy action selection
    if rand < epsilon
        action = randi(length(actionSpace));
    else
        [~, action] = max(Q(context, :));
    end
    
    % Calculate reward based on the non-linear function
    reward = rewardFunction(context, action);
    % Map to binary reward (e.g., click if reward > 0.5)
    binaryReward = reward > 0.5;
    
    % Update Q-values
    Q(context, action) = Q(context, action) + alpha * (binaryReward - Q(context, action));
end

% Display the learned Q-values
disp('Learned Q-values:');
disp(Q);

% Visualization
figure;

% Subplot 1: Ground Truth Reward Function
subplot(1, 2, 1);
[X, Y] = meshgrid(contextSpace, actionSpace);
Z = rewardFunction(X, Y);
surf(X, Y, Z);
title('Ground Truth Reward Function');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Reward');

% Subplot 2: Learned Q-values
subplot(1, 2, 2);
surf(contextSpace, actionSpace, Q');
title('Learned Q-values');
xlabel('Context (Age Group)');
ylabel('Action (Article)');
zlabel('Q-value');

% Use Q-values for prediction (action selection)
% Example prediction for a new user in age group 3
newContext = 3;
[~, bestAction] = max(Q(newContext, :));
disp(['For age group ', num2str(newContext), ', show article ', num2str(bestAction)]);
