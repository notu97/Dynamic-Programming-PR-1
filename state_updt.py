#In[]
import numpy as np
import gym
from utils import *
# import numpy as np

# def state_update(X_now,u_now):
#     if (u_now==0):
#         X_now[0:2]=X_now[0:2]+X_now[2:4]
#         # print(X_now)
#     elif(u_now==1):
#         X_now[2:4]= np.array([[0,-1],[1,0]]) @ X_now[2:4] #np.array([-1,0])
#     elif(u_now==2):
#         X_now[2:4]= np.array([[0,1],[-1,0]]) @ X_now[2:4] #np.array([1,0])
#     elif(u_now==3):
#         print("Key picked")
#     elif(u_now==4):
#         print("Unlock Door")
#         X_now[4]=1
#     return(X_now)


# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

Action=np.array([MF,TL,TR,PK,UD])

#In[]
def value_iteration(env,V_state, max_itr,empty_grids,Action):
    V_state=np.zeros((4*empty_grids,1))
    # state_reward=
    for i in range(max_itr):
        for state in range(len(V_state)): 
            action_value=[]
            #------ move fwd
            cost_MF,done=step(env,MF)
            env.agent_pos
            step(env,TR)    
            step(env,TR)
            step(env,MF)
            step(env,TR)
            step(env,TR)
            
            cost_TL,done=step(env,TL)
            step(env,TR)

            cost_TR,done=step(env,TR)
            step(env,TL)
            V_state[state]=-1
            
            
            for act in Action:
                state_val=0
                
                reward= step(env,act)
                plot_env(env)
                

                 
                plot_env(env)   


# In[]
def doorkey_problem(env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq
#In[]
# def value_iteration(world_grid,V,A,threshold=0.0001):
#     for state in len(V.T): # iterate over States
#         exp_v=0
#         exp_r=0
#         for a in A:
#In[]
# ----------------------illegal move check----------------------

# a=env.agent_pos
# step(env,MF)
# plot_env(env)
# if (a[0]==env.agent_pos[0] and a[1]==env.agent_pos[1]):
#     print("illegal")
#     



#In[]

def main():
    # env_path = './envs/example-8x8.env'
    env_path = './envs/doorkey-8x8-direct.env'
    env, info = load_env(env_path) # load an environment
    
    # step(env,TL)
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.int16)
    # index= np.where(world_grid>=2) and np.where(world_grid<5)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= 10000
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0
    world_grid[np.where(world_grid==1)]=-1
    # print(step_cost(MF))
    l=len(np.where(world_grid<=0)[0])+1
    no_states= l*5 # no of states
    print(no_states)

    # -------------Current pose to key-------------------------------------------
    V=np.zeros((l,5))
    probable_states(env)


    print(world_grid)
    plot_env(env)
    print(info)


    #----------------------------------------------------------------------------
    seq = doorkey_problem(env) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save

#In[]
if __name__ == '__main__':
    # example_use_of_gym_env()
    main()


# %%
