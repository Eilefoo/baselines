#!/usr/bin/env python

import sys
import os
import rospy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1 as tfc
from baselines.a2c.utils import ortho_init
from baselines.common.rotors_wrappers import RotorsWrappers
from baselines.dagger.pid import PID
from baselines.dagger.Actor_model import Actor_model
#from baselines.dagger.buffer import Buffer
import random
from math import floor

from datetime import datetime
from tensorboardX import SummaryWriter
import timeit
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from collections import deque
from planner_msgs.srv import planner_search, planner_searchRequest
from voxblox_msgs.srv import FilePath
from geometry_msgs.msg import Pose

batch_size = 8
nb_training_epoch = 50
steps = 32
dagger_itr = 2
dagger_buffer_size = 40000
gamma = 0.99 # Discount factor for future rewards
tau = 0.001 # Used to update target networks
buffer_capacity=50000 # unused now!
stddev = 0.1
save_path = "/home/eilefoo/models/dagger_multiple_heads"



class Logger:
    def __init__(self):
        self.logdir = "logs/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.info_returns = []
        self.training_epochs = nb_training_epoch #we know that for every set of epochs, there will be one value for percentage collision, reach goal and timeout
        self.current_training_round = 0
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)


    def save_performance_stats(self):
        performance_statistics = np.array(self.info_returns)
        # summation_vector = np.sum(performance_statistics,axis=1)
        # performance_statistics = performance_vector/summation_vector[:,None] #Now each element is a percentage
        performance_statistics.tofile(self.logdir+'/dagger_performance_stats.csv',sep=',')

    def save_model_parameters(self,model,env):
        file1 = open(self.logdir+"/network_and_env_training_summary.txt","w")
        Data_to_save = f'Batch size: {batch_size}\nSteps: {steps}\nNumber of training epochs: {nb_training_epoch}\nDagger iterations: {dagger_itr}\nDagger buffer size: {dagger_buffer_size}\nGamma: {gamma}\nTau: {tau}\nStd. dev.: {stddev} Shape of obs space: {env.ob_robot_state_shape}\nShape of pc:{env.pcl_latent_dim}'
        stringlist = []
        model.summary(line_length=120,print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        file1.writelines(Data_to_save+model_summary)
        file1.close()




class MultipleHeadActor(keras.Model):


    def set_use_left_action(self,action_bool):
        self.use_left_action = action_bool

    #@tf.function
    def train_step(self, data):

        x_true, y_true = data #Both these variables are iterators!!! 
        # x_true_left, x_true_right = tf.split(x_true, num_or_size_splits=2)
        # y_true_left, y_true_right = tf.split(y_true, num_or_size_splits=2)
        print("Inside train_step: \n")
        with tf.GradientTape() as tape:

            y_pred = self(x_true, training=True) #Forward pass

            # if(self.use_left_action == True):
            #     y_pred = [y_pred[0],y_pred[2]]
            # else:
            #     y_pred = [y_pred[1],y_pred[2]]

            print("y_pred: ", y_pred, "\ny_true: ", y_true, "\nIs tensor? ", keras.backend.is_keras_tensor(y_pred[1]), "x_true: ", x_true )
            #print(type(sess.run((y_pred))))
            total_loss = self.total_loss(y_pred, y_true)

        training_vars = self.trainable_variables
        gradients = tape.gradient(total_loss,training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))
        print("\nApply gradients")

        self.compiled_metrics.update_state(y_true, y_pred)

        return {"loss ": total_loss}
    
    def total_loss(self, y_pred, y_true): # y true [None, 3], y_pred = (     )
        # zero = tf.constant([0.0])
        # mask = tf.math.greater(y_true[...,1],zero)
        
        # y_pred_right = y_pred[1][mask]
        # y_pred_left = y_pred[0][~mask]


        # action_loss = tf.losses.mse(y_pred_right,y_true[mask]) + tf.losses.mse(y_pred_left,y_true[~mask])
        # print("\nin this part of the loss function, y_pred: ", y_pred)

        # y_pred_dir = y_pred[2]
        # print("Right before the direction loss calculation, \ny_pred_dir= ", y_pred_dir[:], "\nmask: ", mask,"\n\n")
        
        # direction_loss = keras.losses.binary_crossentropy(mask,y_pred_dir[:])
        # print("Getting closer to the end now")
        # total_loss = action_loss + direction_loss
        y_true_acc, y_true_dir = tf.split(y_true, [3,1],axis=1)
        y_true_dir_bool = tf.cast(y_true_dir, tf.bool)
        y_pred_left_acc = y_pred[0]
        y_pred_right_acc = y_pred[1]
        y_pred_dir = y_pred[2]
        #  = tf.split(y_pred, [3,3,1],axis=1)
        print("\n\nMSE loss, y_true: ", y_true_acc)

        y_pred_acc = tf.where(y_true_dir_bool,y_pred_left_acc,y_pred_right_acc)
        action_loss = tf.losses.mse(y_pred_acc, y_true_acc)
        
        print("\nRight before the direction loss calculation, \ny_true= ",y_true_dir, "\npred: ", y_pred_dir,"\n\n")
        direction_loss = keras.losses.binary_crossentropy(y_true_dir,y_pred_dir)
        
        total_loss = action_loss + direction_loss
        return total_loss 
        

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
            # if (self.cnt == 100):
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
    output_layer_left_action = tf.keras.layers.Dense(units=nb_actions,
                                        name='output_left_action',
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(backbone.outputs[0])
    output_layer_right_action = tf.keras.layers.Dense(units=nb_actions,
                                        name='output_right_action',
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(backbone.outputs[0])
    output_layer_predictor =tf.keras.layers.Dense(units=1,
                                        name='output_predictor',
                                        activation=tf.keras.activations.sigmoid,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(backbone.outputs[0]) 
        
    multiple_head_actor = MultipleHeadActor(inputs=[backbone.inputs], outputs=[output_layer_left_action, output_layer_right_action, output_layer_predictor ], name='actor_net')
    #multiple_head_actor = MultipleHeadActor()
    multiple_head_actor.compile(optimizer= tf.keras.optimizers.Adam(lr=1e-4))
    return multiple_head_actor

# def build_critic_model(input_shape):
#     x_input = tf.keras.Input(shape=input_shape)
#     h = x_input
#     h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
#                                name='mlp_fc1', activation=tf.keras.activations.tanh)(h)
#     h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
#                                name='mlp_fc2', activation=tf.keras.activations.tanh)(h)
#     h = tf.keras.layers.Dense(units=1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
#                                name='output')(h)

#     model = tf.keras.Model(inputs=[x_input], outputs=[h])

#     #optimizer = tf.keras.optimizers.Adam(lr=1e-4)

#     # model.compile(loss='mse',
#     #               optimizer=optimizer,
#     #               metrics=['mse'])
#     return model

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

def load_map():
    #rosservice call /gbplanner_node/load_map "file_path: '/home/eilefoo/reinforcement_learning_ws/src/rmf_sim/voxblox_simple_maze.tsdf'"
    try:
        map_load_service = rospy.ServiceProxy('/gbplanner_node/load_map',load_map)
        response = map_load_service("file_path: '/home/eilefoo/reinforcement_learning_ws/src/rmf_sim/voxblox_simple_maze.tsdf'")
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == '__main__':
    #print('sys.argv:', sys.argv)    
    print("Started dagger.py inside the main \n\n\n\n\n\n\n")

    play = False
    load_path = None
    save_path = None
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
    # Initialize
    env = RotorsWrappers()
    planner_service = rospy.ServiceProxy('/gbplanner/search', planner_search)    
    map_load_service = rospy.ServiceProxy('/gbplanner_node/load_map',FilePath)

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
    logger.save_model_parameters(actor, env)

    episode_rew_queue = deque(maxlen=10)

    num_iterations = 0

    max_mean_return = -100000
    if play:
        obs = env.reset()
        reward_sum = 0.0
        itr = 0
        rospy.sleep(0.5)

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
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0.0
                    obs = env.reset()
                    rospy.sleep(0.5)

            mean_return = np.mean(episode_rew_queue)
            print('Episode done ', 'itr ', itr, 'mean return', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            itr += 1 

    else: # training mode
        robot_left_state_all = np.zeros((1,env.ob_robot_state_shape))
        robot_right_state_all = np.zeros((1,env.ob_robot_state_shape))
        latent_left_pcl_all = np.zeros((1,env.pcl_latent_dim))
        latent_right_pcl_all = np.zeros((1,env.pcl_latent_dim))

        actions_left_all = np.zeros((0, nb_actions+1))  
        actions_right_all = np.zeros((0, nb_actions+1))
        rewards_left_all = np.zeros((0, ))
        rewards_right_all = np.zeros((0, ))

        obs_list = []
        augmented_obs_list = []
        action_list = []
        action_left_list = []
        action_right_list = []
        reward_list = []
        latent_pc_list = []
        num_iterations += 1
        map_load_service('/home/eilefoo/maps/box2_more_stoned.tsdf')


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
            print("action: ",action)
            if (action[0][1] > 0):
                direction = [1]
                action_left_with_dir = np.concatenate([action[0],direction])
                action_list.append(np.array([action_left_with_dir]))
            
            else:
                direction = [0]
                action_right_with_dir = np.concatenate([action[0],direction])
                action_list.append(np.array([action_right_with_dir]))

            #print("Here are the actions * action_space.high: ", action * env.action_space.high)
            #print("Applied action: ", action*env.action_space.high)
            obs, reward, done, _ = env.step(action * env.action_space.high)
            #print("The distance to goal at this time step: ", obs[0:3])
            #print("The latent point cloud at this t step:  ", latest_pc)
            
            reward_list.append(np.array([reward]))
            print("Iteration: ", i)
            if done:
                obs = env.reset()
                augmented_obs = env.get_augmented_obs()
                rospy.sleep(0.3)
        env.pause()

        print('Packing data into arrays...')
        for obs, act, rew, pc in zip(obs_list, action_list, reward_list, latent_pc_list):
            #print("Robot_state: ", robot_state, "Shape of robot_state: ", np.shape(robot_state), "\npcl_feature: ", pcl_feature, "Shape of pcl: ", np.shape(pcl_feature),
            #"\nAction: ", act, "Shape of actions: ", np.shape(act)) sprint("Actions_all: ", actions_all)
            #print("Action in the packing of data: ",act)
            if(act[0][3] == 1):
                robot_state = obs
                pcl_feature = pc
                robot_left_state_all = np.concatenate([robot_left_state_all, robot_state], axis=0)
                latent_left_pcl_all = np.concatenate([latent_left_pcl_all, pcl_feature], axis=0)
                actions_left_all = np.concatenate([actions_left_all, act], axis=0)
                rewards_left_all = np.concatenate([rewards_left_all, rew], axis=0)
            
            elif(act[0][3] == 0):
                robot_state = obs
                pcl_feature = pc
                robot_right_state_all = np.concatenate([robot_right_state_all, robot_state], axis=0)
                latent_right_pcl_all = np.concatenate([latent_right_pcl_all, pcl_feature], axis=0)
                actions_right_all = np.concatenate([actions_right_all, act], axis=0)
                rewards_right_all = np.concatenate([rewards_right_all, rew], axis=0)

            else:
                print("ERROR: the action direction should be either 0 or 1 but it is: ", act[3])

        robot_left_state_all = np.delete(robot_left_state_all, 0,0)
        latent_left_pcl_all = np.delete(latent_left_pcl_all, 0,0)
        robot_right_state_all = np.delete(robot_right_state_all, 0,0)
        latent_right_pcl_all = np.delete(latent_right_pcl_all, 0,0)
        
        #print("\naction left all: ", actions_left_all)
        #print("\npcl all: ", latent_pcl_all)        

        # print("Shape of robot state: ", np.shape(robot_state_all), "\nShape of pc: ", np.shape(latent_pcl_all), "\nShape of actions: ", np.shape(actions_all))
        # concatenated_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
        # print("Shape of the concatenated list: ", np.shape(concatenated_input))
        # print("robot_state_all: ", robot_state_all)
        # print("Latent_pc: ", latent_pcl_all)
        # print("The concatenated list: ", concatenated_input)
        # print("Shape of one element of concat_input: ", concatenated_input[0].shape, "One elemtent of concaten_: ", concatenated_input[0])
        
        input_left_data = np.concatenate((robot_left_state_all, latent_left_pcl_all), axis=1)
        x_y_left_data = np.concatenate((input_left_data, actions_left_all), axis=1)
        input_right_data = np.concatenate((robot_right_state_all, latent_right_pcl_all), axis=1)
        x_y_right_data = np.concatenate((input_right_data, actions_right_all), axis=1)
        #print("x_y_left side: ", x_y_right_data.shape)
        x_y_both_sides = np.concatenate((x_y_left_data, x_y_right_data), axis=0)
        #print("x_y_both sides: ", x_y_both_sides.shape)
        np.random.shuffle(x_y_both_sides)
        shuffled_both_sides = x_y_both_sides
        #print("shuffled space: ", shuffled_both_sides[:,36:40])
        
        actor.fit( shuffled_both_sides[:,0:36], shuffled_both_sides[:,36:40], batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=True,initial_epoch=logger.current_training_round*logger.training_epochs,callbacks=[logger.tensorboard_callback])
        output_file = open('results.txt', 'w')
        logger.current_training_round += 1

        # if(len(actions_left_all)>0):
        #     print("traning left side \n")
        #     actor.set_use_left_action(True)

        #     actor.fit(np.concatenate((robot_left_state_all, latent_left_pcl_all), axis=1), actions_left_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=True)
        #     output_file = open('results.txt', 'w')
        
        # print("Finished training the left side \n\n\n")
        # if(len(actions_right_all)>0):
        #     actor.set_use_left_action(False)
        #     print("traning right side \n")  
        #     actor.fit(np.concatenate((robot_right_state_all, latent_right_pcl_all), axis=1), actions_right_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=True)
        #     output_file = open('results.txt', 'w')
            
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        env.unpause()
        print("\nGoing into Dagger mode \n")
        # Aggregate and retrain actor network
        dagger_buffer_cnt = 0
        for itr in range(dagger_itr):
            result_statistic = np.zeros(3)

            obs_list = []   
            augmented_obs_list = []
            action_list = []
            reward_list = []
            latent_pc_list = []

            robot_left_state_all = np.zeros((1,env.ob_robot_state_shape))
            latent_left_pcl_all = np.zeros((1,env.pcl_latent_dim))
            robot_right_state_all = np.zeros((1,env.ob_robot_state_shape))
            latent_right_pcl_all = np.zeros((1,env.pcl_latent_dim))
            teacher_actions_left_all = np.zeros((0, nb_actions+1))
            rewards_left_all = np.zeros((0, ))
            teacher_actions_right_all = np.zeros((0, nb_actions+1))
            rewards_right_all = np.zeros((0, ))


            obs = env.reset()

            reward_sum = 0.0
            rospy.sleep(0.3)
            print("\n\nDagger iteration: ", itr, " of ", dagger_itr)
            for i in range(steps):
                #print('obs:', obs)
                latest_pcl = env.get_latest_pcl_latent()

                concatenated_input = np.concatenate((obs, latest_pcl), axis=0)
                concatenated_input = np.reshape(concatenated_input,(1,env.ob_robot_state_shape + env.pcl_latent_dim))
            
                #start = timeit.default_timer()
                pred_action = actor(concatenated_input, training=False)  # assume symmetric action space (low = -high)
                print("Actor actions: ", action*env.action_space.high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)
                
                if(pred_action[2]> 0.5):
                    action = pred_action[0]
                elif(pred_action[2] <= 0.5):
                    action = pred_action[1]
                
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
                    #Record 
                    if info['status'] == "reach goal":
                        result_statistic[0] +=1
                    if info['status'] == "collide":
                        result_statistic[1] +=1
                    if info['status'] == "timeout":
                        result_statistic[2] +=1

                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0
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
                if (len(actions_left_all) + len(actions_right_all) < dagger_buffer_size):
                    print("Inside if")
                    if(teacher_action[0][1] > 0):
                        robot_left_state_all = np.concatenate([robot_left_state_all, robot_state], axis=0)
                        latent_left_pcl_all = np.concatenate([latent_left_pcl_all, pcl_feature], axis=0)
                        teacher_left_action_list = []
                        teacher_action_left_with_dir = np.concatenate([teacher_action[0],[1]])
                        teacher_left_action_list.append(teacher_action_left_with_dir)
                        teacher_actions_left_all = np.concatenate((teacher_actions_left_all, teacher_left_action_list), axis=0)
                    
                    elif (teacher_action[0][1] <= 0):
                        robot_right_state_all = np.concatenate([robot_right_state_all, robot_state], axis=0)
                        latent_right_pcl_all = np.concatenate([latent_right_pcl_all, pcl_feature], axis=0)
                        teacher_right_action_list = []
                        teacher_action_right_with_dir = np.concatenate([teacher_action[0],[0]])

                        teacher_right_action_list.append(teacher_action_right_with_dir)
                        teacher_actions_right_all = np.concatenate((teacher_actions_right_all, teacher_right_action_list), axis=0)

                    else:
                        print("ERROR: teacher action[1] is neither above or below zero, but: ", teacher_action[0])

                else: # buffer is full

                    print("Inside else")
                    dagger_buffer_cnt += 1
                    if (dagger_buffer_cnt == dagger_buffer_size):
                        print('reset dagger_buffer_cnt')
                        dagger_buffer_cnt = 0

            #Removing the initialization elements
            robot_left_state_all = np.delete(robot_left_state_all, 0,0) 
            latent_left_pcl_all = np.delete(latent_left_pcl_all, 0,0)
            robot_right_state_all = np.delete(robot_right_state_all, 0,0) 
            latent_right_pcl_all = np.delete(latent_right_pcl_all, 0,0)
            
            # print("Shape of robot state: ", np.shape(robot_state_all), "\nShape of pc: ", np.shape(latent_pcl_all), "\nShape of actions: ", np.shape(actions_all))
            # concatenated_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
            # print("Shape of the concatenated list: ", np.shape(concatenated_input))
            # print("robot_state_all: ", robot_state_all)
            # print("Latent_pc: ", latent_pcl_all)
            # print("The concatenated list: ", concatenated_input)
            # print("Shape of one element of concat_input: ", concatenated_input[0].shape, "One element of concated_input: ", concatenated_input[0])
            # train actor
            #concat_input = np.concatenate((robot_state_all, latent_pcl_all), axis=1)
            #print(concat_input)


            input_left_data = np.concatenate((robot_left_state_all, latent_left_pcl_all), axis=1)
            x_y_left_data = np.concatenate((input_left_data, teacher_actions_left_all), axis=1)
            input_right_data = np.concatenate((robot_right_state_all, latent_right_pcl_all), axis=1)
            x_y_right_data = np.concatenate((input_right_data, teacher_actions_right_all), axis=1)

            x_y_both_sides = np.concatenate((x_y_left_data, x_y_right_data), axis=0)

            np.random.shuffle(x_y_both_sides)
            shuffled_both_sides = x_y_both_sides

            while len(shuffled_both_sides) % batch_size != 0: #This is to ensure the length of the data is divisible by the batch size
                print("The length of shuffled both sides ",len(shuffled_both_sides))
                rand_index = floor(random.uniform(0,len(shuffled_both_sides)))
                shuffled_both_sides = np.concatenate((shuffled_both_sides, [shuffled_both_sides[rand_index]]), axis=0)

            # if(len(actions_left_all)>0):
            #     print("traning left side \n")
            #     actor.set_use_left_action(True)

            #     actor.fit(np.concatenate((robot_left_state_all, latent_left_pcl_all), axis=1), teacher_actions_left_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=True)
            #     output_file = open('results.txt', 'w')
        
            # print("Finished training the left side \n\n\n")
            # if(len(actions_right_all)>0):
            #     actor.set_use_left_action(False)
            #     print("traning right side \n")  
            #     actor.fit(np.concatenate((robot_right_state_all, latent_right_pcl_all), axis=1), teacher_actions_right_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=True)
            #     output_file = open('results.txt', 'w')
            
            print("Dagger training for actor")
            actor.fit(shuffled_both_sides[:,0:56], shuffled_both_sides[:,56:60],
                            batch_size=batch_size,
                            epochs=nb_training_epoch,
                            shuffle=True, verbose=True,initial_epoch=logger.current_training_round*logger.training_epochs,callbacks=[logger.tensorboard_callback])
                            #validation_split=0.2, verbose=0,
            logger.current_training_round += 1
        actor.save_weights('dagger_pcl.h5')
        if (save_path != None):
            #actor.save('dagger_actor_pcl', include_optimizer=False) # should we include optimizer?
            print('save weights to file:', save_path)
            actor.save_weights(save_path + '/dagger_pcl_21_04_new_pc_model128_128_64_half_latent_50_multiple_heads.h5')
    
    #Save the dagger performance to file
    logger.save_performance_stats()

