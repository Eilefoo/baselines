#!/usr/bin/env python

import sys
import rospy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.a2c.utils import ortho_init
from baselines.common.rotors_wrappers import RotorsWrappers
from baselines.dagger.pid import PID
#from baselines.dagger.buffer import Buffer
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import timeit
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from collections import deque
from planner_msgs.srv import planner_search, planner_searchRequest
from std_srvs.srv import Empty
from voxblox_msgs.srv import FilePath
from geometry_msgs.msg import Pose

batch_size = 32
steps = 300000
nb_training_epoch = 50
dagger_itr = 0
dagger_buffer_size = 40000
gamma = 0.99 # Discount factor for future rewards
tau = 0.001 # Used to update target networks
buffer_capacity=50000 # unused now!
stddev = 0.1
save_path = "/home/eilefoo/models/dagger"

class Logger:
    def __init__(self):
        self.logdir = "../logs/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.info_returns = []
        self.training_epochs = nb_training_epoch #we know that for every set of epochs, there will be one value for percentage collision, reach goal and timeout
        self.current_training_round = 0


    def save_performance_stats(self):
        performance_statistics = self.info_returns
        summation_vector = np.sum(performance_statics,axis=1)
        performance_statistics = performance_vector/summation_vector[:,None] #Now each element is a percentage
        performance_statistics.to_file(self.logdir+'/dagger_performance_stats.csv',sep=',')





class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, nb_obs))
        self.action_buffer = np.zeros((self.buffer_capacity, nb_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, nb_obs))

        self.cnt = 0

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = actor(next_state_batch) # from dagger, no target actor
            y = reward_batch + gamma * target_critic(tf.concat([next_state_batch, tf.cast(target_actions, dtype='float64')], axis=-1))
            critic_value = critic(tf.concat([state_batch, tf.cast(action_batch, dtype='float64')], axis=-1))
            #y = reward_batch + gamma * target_critic([next_state_batch, tf.cast(target_actions, dtype='float64')])
            #critic_value = critic([state_batch, tf.cast(action_batch, dtype='float64')])             
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            # self.cnt = self.cnt + 1
            # if (self.cnt == 100):  u  
            #     self.cnt = 0
            #     print('y:', y)
            #     print('critic_value:', critic_value)

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

        # with tf.GradientTape() as tape:
        #     actions = actor_model(state_batch)
        #     critic_value = critic_model([state_batch, actions])
        #     # Used `-value` as we want to maximize the value given
        #     # by the critic for our actions
        #     actor_loss = -tf.math.reduce_mean(critic_value)

        # actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # actor_optimizer.apply_gradients(
        #     zip(actor_grad, actor_model.trainable_variables)
        # )
        return critic_loss

def get_teacher_action(planner, controller, augmented_obs, action_space):
    request = planner_searchRequest()
    request.use_current_state = False
    source = Pose()
    source.position.x = augmented_obs[0]
    source.position.y = augmented_obs[1]
    source.position.z = augmented_obs[2]
    source.orientation.x = 0.0
    source.orientation.y = 0.0
    source.orientation.z = 0.0
    source.orientation.w = 1.0
    target = Pose()
    target.position.x = augmented_obs[6]
    target.position.y = augmented_obs[7]
    target.position.z = augmented_obs[8]
    target.orientation.x = 0.0
    target.orientation.y = 0.0
    target.orientation.z = 0.0
    target.orientation.w = 1.0
    request.source = source
    request.target = target
    #print("Request for expert: ", request, "\n\n\n")
    #rospy.wait_for_service(planner)
    response = planner(request)
    path = response.path
    #print("Response from expert: ", response, "\n\n\n")

    if (len(path) > 1):
        waypoint = path[1].position
        vel_setpoint = np.array([waypoint.x - augmented_obs[0], waypoint.y - augmented_obs[1], waypoint.z - augmented_obs[2]])
        if (len(path) > 5): #5
            vel_magnitude = 1.0 / 2
        else:
            vel_magnitude = (0.15 * (len(path) - 1))
        vel_setpoint_norm = np.linalg.norm(vel_setpoint)
        if (vel_setpoint_norm != 0):
            vel_setpoint = vel_setpoint * vel_magnitude / vel_setpoint_norm

        obs = np.array([waypoint.x - augmented_obs[0], waypoint.y - augmented_obs[1], waypoint.z - augmented_obs[2],
                    vel_setpoint[0] - augmented_obs[3], vel_setpoint[1] - augmented_obs[4], vel_setpoint[2] - augmented_obs[5]])
        action = controller.calculate(obs)
        #print("Action: ", action)
        action = tf.clip_by_value(action, -1.0, 1.0)
        action = np.array([action])
    elif (len(path) == 1):
        action = np.array([[0.0, 0.0, 0.0]])
    else: # cannot find a path
        action = np.array([])        
    return action

# def pcl_encoder(input_shape):
#     print('pcl_encoder input shape is {}'.format(input_shape))
#     inputs = tf.keras.Input(shape=input_shape[0] * input_shape[1] * input_shape[2])
#     x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], input_shape[2]))(inputs)
#     x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 3), padding='same', name='pool1')(x)
#     x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 3), padding='same', name='pool2')(x)
#     # Generate the latent vector
#     latent = tf.keras.layers.Flatten()(x)
#     encoder = tf.keras.Model(inputs, latent, name='encoder')
#     #encoder.summary()
#     return encoder

# build network
def build_backbone(layer_input_shape):
    robot_state_input = tf.keras.Input(shape=layer_input_shape)
    # FC layers
    h1 = tf.keras.layers.Dense(units=128, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='fc1', activation='relu')(robot_state_input)
    h2 = tf.keras.layers.Dense(units=128, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='fc2', activation='relu')(h1)
    h3 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='fc3', activation='relu')(h2)
    backbone_model = tf.keras.Model(inputs=robot_state_input, outputs=[h3], name='backbone_net')
    return backbone_model

def build_actor_model(ob_robot_state_shape, ob_pcl_shape, nb_actions):
    #Constructing placeholders
    robot_state_placeholder = np.zeros((1,ob_robot_state_shape))
    pcl_placeholder = np.zeros((1,ob_pcl_shape))
    #print("Robot placeholder: ", robot_state_placeholder[0])
    layer_input_placeholder = np.concatenate((robot_state_placeholder, pcl_placeholder), axis=1)
    print("Input shape ", layer_input_placeholder, "Shape of input shape: ", np.shape(layer_input_placeholder))

    # input layer
    backbone = build_backbone(layer_input_shape=np.shape(layer_input_placeholder))
    # output layer
    output_layer = tf.keras.layers.Dense(units=nb_actions,
                                        name='output',
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(backbone.outputs[0])
    model = tf.keras.Model(inputs=[backbone.inputs], outputs=[output_layer], name='actor_net')
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

def build_critic_model(input_shape):
    x_input = tf.keras.Input(shape=input_shape)
    h = x_input
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc1', activation=tf.keras.activations.tanh)(h)
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc2', activation=tf.keras.activations.tanh)(h)
    h = tf.keras.layers.Dense(units=1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                               name='output')(h)

    model = tf.keras.Model(inputs=[x_input], outputs=[h])

    #optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # model.compile(loss='mse',
    #               optimizer=optimizer,
    #               metrics=['mse'])
    return model

# def build_critic_model(input_shape):
#     # State as input
#     state_input = layers.Input(shape=(6))
#     state_out = layers.Dense(16, activation="relu")(state_input)
#     state_out = layers.BatchNormalization()(state_out)
#     state_out = layers.Dense(32, activation="relu")(state_out)
#     state_out = layers.BatchNormalization()(state_out)

#     # Action as input
#     action_input = layers.Input(shape=(3))
#     action_out = layers.Dense(32, activation="relu")(action_input)
#     action_out = layers.BatchNormalization()(action_out)

#     # Both are passed through seperate layer before concatenating
#     concat = layers.Concatenate()([state_out, action_out])

#     out = layers.Dense(64, activation="relu")(concat)
#     out = layers.BatchNormalization()(out)
#     out = layers.Dense(64, ac0.6tivation="relu")(out)
#     out = layers.BatchNormalization()(out)
#     outputs = layers.Dense(1)(out)

#     # Outputs single value for give state-action
#     model = tf.keras.Model([state_input, action_input], outputs)

#     return model

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

def load_map(load_path):
    #rosservice call /gbplanner_node/load_map "file_path: '/home/eilefoo/reinforcement_learning_ws/src/rmf_sim/voxblox_simple_maze.tsdf'"
    try:
        response = map_load_service(load_path)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == '__main__':
    #print('sys.argv:', sys.argv)    
    print("Started dagger.py inside the main \n\n\n\n\n\n\n")

    play = False
    load_path = None
    save_path = None
    world_scramble = False
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--play':
            play = True
            print('PLAY')
            continue
        index = sys.argv[i].find('--load_path')
        if (index != -1):
            load_path = sys.argv[i][12:]
            continue
        index = sys.argv[i].find('--save_path')
        if (index != -1):
            save_path = sys.argv[i][12:]
            continue
        index = sys.argv[i].find('--world_scramble')
        if (index != -1):
            world_scramble = True
            continue
        
    # Initialize
    env = RotorsWrappers()
    planner_service = rospy.ServiceProxy('/gbplanner/search', planner_search)    
    map_load_service = rospy.ServiceProxy('/gbplanner_node/load_map',FilePath)
    map_clear_service = rospy.ServiceProxy('/gbplanner_node/clear_map',Empty)


    # Limiting GPU memory growth: https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #tf.keras.backend.set_floatx('float32')|
    save_path = "/home/eilefoo/models/dagger"

    #Initialize progress saver
    logger = Logger()
    
    pid = PID([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    nb_actions = env.action_space.shape[-1]
    nb_obs = env.observation_space.shape[-1]
    print("Shape of obs space: ", env.ob_robot_state_shape, "\nShape of pc: ", env.pcl_latent_dim)
    # actor
    actor = build_actor_model(ob_robot_state_shape=env.ob_robot_state_shape, ob_pcl_shape=env.pcl_latent_dim, nb_actions=nb_actions)
    actor.summary()
    if (load_path != None): 
        print('load model from path:', load_path)
        actor.load_weights(load_path)

    writer = SummaryWriter(comment="-rmf_dagger_pcl_latent")

    episode_rew_queue = deque(maxlen=10)

    env_reset_counter = 0 
    num_iterations = 0

    max_mean_return = -100000
    if play:
        obs = env.reset()
        reward_sum = 0.0
        itr = 0
        rospy.sleep(0.5)
        env.pause()
        tsdf_filename = '/home/eilefoo/maps/random_generated_maps/random_%i.tsdf' % (i)
        scrambler_bool = env.scramble_world()

        env_reset_counter = 0
        env.unpause()
        while True:
            for i in range(steps):
                #print('obs:', obs)
                rospy.sleep(0.1)
                latest_pcl = env.get_latest_pcl_latent()

                concatenated_input = np.concatenate((obs, latest_pcl), axis=0)
                concatenated_input = np.reshape(concatenated_input,(1,6 + env.pcl_latent_dim))

                #start = timeit.default_timer()
                action = actor(concatenated_input, training=False)  # assume symmetric action space (low = -high)
                print("\nPlaying, action value: ", action*env.action_space.high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)

                action = tf.clip_by_value(action, -1.0, 1.0)

                obs, reward, done, _ = env.step(action * env.action_space.high)
                reward_sum += reward

                if done is True:
                    env_reset_counter = env_reset_counter +1
                    if (env_reset_counter > 1): 
                        env.pause()

                        scrambler_bool = env.scramble_world()
                        env_reset_counter = 0
                        env.unpause()

                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0.0
                    obs = env.reset()
                    rospy.sleep(0.5)

            mean_return = np.mean(episode_rew_queue)
            print('Episode done ', 'itr ', itr, 'mean return', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            itr += 1 

    else: # training mode
        robot_state_all = np.zeros((1,env.ob_robot_state_shape))
        latent_pcl_all = np.zeros((1,env.pcl_latent_dim))

        actions_all = np.zeros((0, nb_actions))
        rewards_all = np.zeros((0, ))

        obs_list = []
        augmented_obs_list = []
        action_list = []
        reward_list = []
        latent_pc_list = []
        num_iterations += 1
        if(world_scramble):
            tsdf_filename = "/home/eilefoo/maps/random_generated_maps/random_1.tsdf"
            env.pause()
            scrambler_bool = env.scramble_world()
            world_converter_bool = env.world_to_tsdf_converter(tsdf_filename)
            map_clear_service()
            map_load_service(tsdf_filename)
            env.unpause()
            env_reset_counter = 0
        # Collect data with expert in first iteration
        obs = env.reset()
        augmented_obs = env.get_augmented_obs()
        print("This is the first observation: ", obs)
        rospy.sleep(0.3)
        print('Collecting data...')
        for i in range(steps):
            rospy.sleep(0.1)
            counter = 0
            augmented_obs = env.get_augmented_obs()            
            action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)
            latest_pcl = env.get_latest_pcl_latent()
            #print("This is the action from the expert, ", action, "The length is: ", len(action))
            while len(action) == 0:
                print('no expert path, action is: ', action)
                #obs = env.reset()
                #print("The observations after reseting: \n\n\n\n", augmented_obs, "\n\n\n\n")
                augmented_obs = env.get_augmented_obs()
                counter = counter + 1
                rospy.sleep(0.1)
                latest_pcl = env.get_latest_pcl_latent()
                action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)                
                if (counter > 7):
                    obs = env.reset()
                    augmented_obs = env.get_augmented_obs()
                    rospy.sleep(0.3)

            #Saving the data
            obs_list.append(np.array([obs]))
            latent_pc_list.append(np.array([latest_pcl]))
            augmented_obs_list.append(augmented_obs)
            action_list.append(np.array(action))

            #print("Here are the actions * action_space.high: ", action * env.action_space.high)
            #print("Applied action: ", action*env.action_space.high)
            obs, reward, done, _ = env.step(action * env.action_space.high)
            #print("The distance to goal at this time step: ", obs[0:3])
            #print("The latent point cloud at this t step:  ", latest_pc)
            
            reward_list.append(np.array([reward]))
            print("Iteration: ", i)

            if done:
                if(world_scramble):
                    env_reset_counter = env_reset_counter +1
                    if (env_reset_counter > 1): 
                        env.pause()
                        tsdf_filename = '/home/eilefoo/maps/random_generated_maps/random_%i.tsdf' % (i)
                        scrambler_bool = env.scramble_world()
                        world_to_tsdf_bool = env.world_to_tsdf_converter(tsdf_filename)
                        print("The world_converter_bool came back: ", world_to_tsdf_bool)
                        map_clear_service()
                        map_load_service(tsdf_filename)
                        env_reset_counter = 0
                        env.unpause()

                obs = env.reset()
                augmented_obs = env.get_augmented_obs()
                rospy.sleep(0.3)
        env.pause()

        print('Packing data into arrays...')
        for obs, act, rew, pc in zip(obs_list, action_list, reward_list, latent_pc_list):
            robot_state = obs
            pcl_feature = pc
            print("Robot_state: ", robot_state, "Shape of robot_state: ", np.shape(robot_state), "\npcl_feature: ", pcl_feature, "Shape of pcl: ", np.shape(pcl_feature),
            "\nAction: ", act, "Shape of actions: ", np.shape(act))
            print("Actions_all: ", actions_all)
            robot_state_all = np.concatenate([robot_state_all, robot_state], axis=0)
            latent_pcl_all = np.concatenate([latent_pcl_all, pcl_feature], axis=0)
            actions_all = np.concatenate([actions_all, act], axis=0)
            rewards_all = np.concatenate([rewards_all, rew], axis=0)

        robot_state_all = np.delete(robot_state_all, 0,0) 
        latent_pcl_all = np.delete(latent_pcl_all, 0,0)
        
        #print("\nRobot state all: ", robot_state_all)
        #print("\npcl all: ", latent_pcl_all)        

        # print("Shape of robot state: ", np.shape(robot_state_all), "\nShape of pc: ", np.shape(latent_pcl_all), "\nShape of actions: ", np.shape(actions_all))
        # concatenated_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
        # print("Shape of the concatenated list: ", np.shape(concatenated_input))
        # print("robot_state_all: ", robot_state_all)
        # print("Latent_pc: ", latent_pcl_all)
        # print("The concatenated list: ", concatenated_input)
        # print("Shape of one element of concat_input: ", concatenated_input[0].shape, "One elemtent of concaten_: ", concatenated_input[0])
        # First train for actor network
        actor.fit(np.concatenate((robot_state_all, latent_pcl_all), axis=1), actions_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=False,initial_epoch=logger.current_training_round*logger.training_epochs,callbacks=[logger.tensorboard_callback])
        output_file = open('results.txt', 'w')
        logger.current_training_round += 1
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        env.unpause()
        print("\nGoing into Dagger mode \n")
        # Aggregate and retrain actor network
        dagger_buffer_cnt = 0
        for itr in range(dagger_itr):
            #idx0: reach goal, idx1: collide, idx2: timeout
            result_statistic = np.zeros(3)
            
            obs_list = []
            augmented_obs_list = []
            action_list = []
            reward_list = []
            latent_pc_list = []

            robot_state_all = np.zeros((1,env.ob_robot_state_shape))
            latent_pcl_all = np.zeros((1,env.pcl_latent_dim))
            actions_all = np.zeros((0, nb_actions))
            rewards_all = np.zeros((0, ))
            
            obs = env.reset()

            reward_sum = 0.0
            rospy.sleep(0.3)
            print("\n\nDagger iteration: ", itr, " of ", dagger_itr)
            for i in range(steps):
                #print('obs:', obs)
                #rospy.sleep(0.1)
                latest_pcl = env.get_latest_pcl_latent()

                concatenated_input = np.concatenate((obs, latest_pcl), axis=0)
                concatenated_input = np.reshape(concatenated_input,(1,56))
                #print("Concatenated input shape: ", np.shape(concatenated_input))
            
                #start = timeit.default_timer()
                action = actor(concatenated_input, training=False)  # assume symmetric action space (low = -high)
                #print("Actor actions: ", action*env.action_space.high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)
                #print('action:', action) # action = [[.]]

                action = tf.clip_by_value(action, -1, 1)

                new_obs, reward, done, info = env.step(action * env.action_space.high)
                augmented_obs = env.get_augmented_obs()
                obs = new_obs
                reward_sum += reward
                print("Iteration: ", i)

                obs_list.append(np.array([obs]))
                latent_pc_list.append(np.array([latest_pcl]))
                augmented_obs_list.append(augmented_obs)
                action_list.append(np.array(action))

                if done or i == steps-1:
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0
                    if(world_scramble):
                        env_reset_counter = env_reset_counter +1
                        if (env_reset_counter >=1): 
                            env.pause()
                            tsdf_filename = '/home/eilefoo/maps/random_generated_maps/random_%i.tsdf' % (i)
                            scrambler_bool = env.scramble_world()
                            world_to_tsdf_bool = env.world_to_tsdf_converter(tsdf_filename)
                            print("The world_converter_bool came back: ", world_to_tsdf_bool)
                            map_clear_service()
                            map_load_service(tsdf_filename)
                            env_reset_counter = 0
                            env.unpause()

                    #Record 
                    if info.status == "reach goal":
                        result_statistic[0] +=1
                    if info.status == "collide":
                        result_statistic[1] +=1
                    if info.status == "timeout":
                        result_statistic[2] +=1

                    
                    
                    obs = env.reset()
                    augmented_obs = env.get_augmented_obs()
                    rospy.sleep(0.3) #Wait for pointcloud 
                    continue

            #Saving the Dagger performance statistic
            logger.info_returns.append(result_statistic)

            env.pause()

            mean_return = np.mean(episode_rew_queue)
            print('Episode done ', 'itr ', itr, ',i ', i, 'mean return', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            if (mean_return > max_mean_return):
                max_mean_return = mean_return
                actor.save_weights('dagger_pcl_itr' + str(itr) + '.h5')
            
            print("\n\nRight before packing data \n\n")

            for obs, augmented_obs, pc in zip(obs_list, augmented_obs_list, latent_pc_list):
                teacher_action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)
                #print("Teacher action: ", teacher_action)
                if len(teacher_action) == 0:
                    print('found no expert path. robot state is: ', augmented_obs[0:3], " goal is: ", obs[0:3])
                    continue
                #print("Teacher actions is: ", teacher_action ,"Robot state is: ", augmented_obs[0:3], 'goal state is', obs[0:3])
                robot_state = obs
                pcl_feature = pc
                #print('robot_state:', robot_state)
                if (len(actions_all) < dagger_buffer_size):
                    #print("Inside if")
                    robot_state_all = np.concatenate([robot_state_all, robot_state], axis=0)
                    latent_pcl_all = np.concatenate([latent_pcl_all, pcl_feature], axis=0)
                    actions_all = np.concatenate([actions_all, teacher_action], axis=0)
                else: # buffer is full

                    #print("Inside else")
                    robot_state_all[dagger_buffer_cnt] = robot_state
                    latent_pcl_all[dagger_buffer_cnt] = pcl_feature
                    actions_all[dagger_buffer_cnt] = teacher_action
                    dagger_buffer_cnt += 1
                    if (dagger_buffer_cnt == dagger_buffer_size):
                        print('reset dagger_buffer_cnt')
                        dagger_buffer_cnt = 0

            #Removing the initialization elements
            robot_state_all = np.delete(robot_state_all, 0,0) 
            latent_pcl_all = np.delete(latent_pcl_all, 0,0)
            
            # print("Shape of robot state: ", np.shape(robot_state_all), "\nShape of pc: ", np.shape(latent_pcl_all), "\nShape of actions: ", np.shape(actions_all))
            # concatenated_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
            # print("Shape of the concatenated list: ", np.shape(concatenated_input))
            # print("robot_state_all: ", robot_state_all)
            # print("Latent_pc: ", latent_pcl_all)
            # print("The concatenated list: ", concatenated_input)
            # print("Shape of one element of concat_input: ", concatenated_input[0].shape, "One element of concated_input: ", concatenated_input[0])
            # train actor
            concat_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
            #print(concat_input)
            print("Dagger training for actor")
            actor.fit(concat_input, actions_all,
                            batch_size=batch_size,
                            epochs=nb_training_epoch,
                            shuffle=True, verbose=False,initial_epoch=logger.current_training_round*logger.training_epochs,callbacks=[logger.tensorboard_callback])
                            #validation_split=0.2, verbose=0,
            logger.current_training_round += 1
        actor.save_weights('dagger_pcl.h5')
        if (save_path != None):
            #actor.save('dagger_actor_pcl', include_optimizer=False) # should we include optimizer?
            print('save weights to file:', save_path)
            actor.save_weights(save_path + '/dagger_pcl_04_05_model128_128_64_30latent_old_map12222222222222222h5')
    
    #Save the dagger performance to file
    logger.save_performance_stats()