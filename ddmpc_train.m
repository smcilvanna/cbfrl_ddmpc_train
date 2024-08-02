addpath("~/matlab/com/casadi/");
%%


obs_test_set = [0.1:0.1:2.0]';  % Define Obstacles For Testing

for i = 1:size(obs_test_set,1)
    obs_radius = obs_test_set(i);

    test_params(obs_radius);

end





%% ############## LOCAL FUNCTIONS
%% Sweep Parameter Set For Given Obstacle
function test_params(obs_radius)

    test_param_set = [0.01 :0.01: 1.50]';
    all_data = [];
    
    % obs_radius = 1.0;

    savename = "mpc_cbf_param_sweep_obs_r" + sprintf('%.2f', obs_radius) + ".mat";


    for i = 1:size(test_param_set,1)
        garma = test_param_set(i);
        run_data = ddrob_mpc_cbf_ps(garma, obs_radius);
        all_data = [all_data ; run_data];
        disp(i);
    end

    save(savename);

end