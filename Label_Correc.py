#In[]
import numpy as np
import gym
import math
from utils import *

# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

Action=np.array([MF,TL,TR,PK,UD])

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
def fill_4_cells(cost_grid,loc,val,grid_flag):
    l=loc[0]
    b=loc[1]
    print(l,b)
    # r=cost_grid[l,b]+1
    if cost_grid[l-1,b]!= math.inf and grid_flag[l-1,b]==0:
        cost_grid[l-1,b]=cost_grid[l,b]+1 #Top
        grid_flag[l-1,b]=1
    else:
    #     cost_grid[l-1,b]=math.inf
        grid_flag[l-1,b]=1

    if cost_grid[l+1,b]!= math.inf and grid_flag[l+1,b]==0 :
        cost_grid[l+1,b]=cost_grid[l,b]+1 #Bottom
        grid_flag[l+1,b]=1
    else:
    #     cost_grid[l+1,b]=math.inf
        grid_flag[l+1,b]=1

    if cost_grid[l,b-1]!= math.inf and grid_flag[l,b-1]==0:
        cost_grid[l,b-1]=cost_grid[l,b]+1 #Left
        grid_flag[l,b-1]=1
    else:
    #     cost_grid[l,b-1]=math.inf
        grid_flag[l,b-1]=1

    if cost_grid[l,b+1]!= math.inf and grid_flag[l,b+1]==0:
        cost_grid[l,b+1]=cost_grid[l,b]+1 #Right
        grid_flag[l,b+1]=1
    else:
    #     cost_grid[l,b+1]=math.inf
        grid_flag[l,b+1]=1

    a=min(np.max(cost_grid),cost_grid[l,b]+1)
    print('a------------------------>>> ',a)
    return cost_grid,grid_flag,(cost_grid[l,b]+1)

#In[]
def label_Correction(env,agentPos,cost_grid,goal,grid_flag):

    c=0
    goal=np.roll(goal,1)
    # agentPos=info['init_agent_pos']
    agentPos=np.roll(agentPos,1)
    # print(goal)
    cost_grid,grid_flag,r = fill_4_cells(cost_grid, goal,0, grid_flag)
    
    # for location in np.where(cost_grid==1):
    # x=np.where(cost_grid==1)[0]
    # y=np.where(cost_grid==1)[1]
    print('Cost_grid')
    print(cost_grid)
    print('Grid_flag')
    print(grid_flag)
    
    while c<=r:
        
        q=np.vstack((np.where(cost_grid==r)[0],np.where(cost_grid==r)[1]))
        print(q)
        grid_flag[goal[0],goal[1]]=1
        # print(y)
        for i in range(len(q.T)):
            cost_grid,grid_flag,r= fill_4_cells(cost_grid, q[:,i].T,0, grid_flag)
            # print(r)
            # a=min(np.max(cost_grid),r)
            # print("r=-----------> ",r)
            # cost_grid,grid_flag= fill_4_cells(cost_grid, q[:,1].T,0, grid_flag)
            # cost_grid,grid_flag= fill_4_cells(cost_grid, q[:,2].T,0, grid_flag)
        # print(c)
        c=c+1
    # if( grid_flag[agentPos[0],agentPos[1]] ==0):
    #     print('We need key')
    # else:
    #     print('We dont need key')

    print('Cost_grid')
    print(cost_grid)
    print('Grid_flag')
    print(grid_flag)

    
#In[]
def main():

    # env_path = './envs/example-8x8.env'
    # env_path = './envs/doorkey-5x5-normal.env'
    # env_path = './envs/doorkey-6x6-direct.env'
    # env_path = './envs/doorkey-6x6-normal.env'
    # env_path = './envs/doorkey-6x6-shortcut.env' # Some Problem
    # env_path = './envs/doorkey-8x8-direct.env'
    # env_path = './envs/doorkey-8x8-normal.env'
    env_path = './envs/doorkey-8x8-shortcut.env' # some problem # may be solved

    env, info = load_env(env_path) # load an environment
    goal=info['goal_pos']
    agentPos=info['init_agent_pos']
    keyPos=info['key_pos']
    doorPos=info['door_pos']
    # step(env,TL)
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    # index= np.where(world_grid>=2) and np.where(world_grid<5)
    
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    # print(world_grid)
    grid_flag=np.zeros(np.shape(world_grid))
    # print(grid_flag)

    
    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0
    # cost_grid[info['key_pos'][1],info['key_pos'][0]]=2
    print(cost_grid)

    # V=np.zeros((l,5))
    #------------- When Door closed-----------
    # Cost without door
    label_Correction(env,agentPos,cost_grid,goal,grid_flag)
    # cost_grid[np.where(cost_grid<=0)]=0

    # cost_with_door_closed=cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]

    if(cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ] ==0):
        cost_with_door_closed=math.inf
    else:
        cost_with_door_closed=cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]
    
        # cost_without_door=cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]
    
    #-------------When Door Open-------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    # print(world_grid)
    grid_flag=np.zeros(np.shape(world_grid))
    # print(grid_flag)

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    # init pos to Key
    label_Correction(env,agentPos,cost_grid,keyPos,grid_flag) 
    c1=cost_grid[agentPos[1],agentPos[0]]-1
    print("C1= ",c1)
    # print(world_grid)

    # ------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    # print(world_grid)
    grid_flag=np.zeros(np.shape(world_grid))
    # print(grid_flag)

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,keyPos,cost_grid,doorPos,grid_flag) 
    c2=cost_grid[keyPos[1],keyPos[0]]-1
    print("C2= ",c2)
    #--------------------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    # print(world_grid)
    grid_flag=np.zeros(np.shape(world_grid))
    # print(grid_flag)

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,doorPos,cost_grid,goal,grid_flag) 
    c3=cost_grid[doorPos[1],doorPos[0]]
    print("C3= ",c3)

    cost_with_door_open=c1+c2+c3

    # grid_flag=np.zeros(np.shape(world_grid))
    # cost_grid=world_grid
    # cost_grid[np.where(cost_grid<=0)]=0
    
    # print(cost_grid)
    # print(world_grid)
    # # Init Pos to Key
    # label_Correction(env,agentPos,cost_grid,keyPos,grid_flag) 
    # c1=cost_grid[agentPos[1],agentPos[0]]-1
    # print(c1)

    # cost_grid=world_grid
    # cost_grid[np.where(cost_grid<=0)]=0
    # grid_flag=np.zeros(np.shape(world_grid))
    
    # # Key to Door
    # label_Correction(env,keyPos,cost_grid,doorPos,grid_flag) 
    # c2=cost_grid[agentPos[1],agentPos[0]]-1
    # print(c2)



    # cost_with_door_open= cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]

    print('Cost with DOOR CLosed   ', cost_with_door_closed)
    print('Cost with DOOR Open  ', cost_with_door_open)

    if(cost_with_door_closed>cost_with_door_open):
        print('We need Key')
    else:
        print('No key needed')

    # print(world_grid)
    plot_env(env)
    # print(info)


    #----------------------------------------------------------------------------
    # seq = doorkey_problem(env) # find the optimal action sequence
    # draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save


#In[]
if __name__ == '__main__':
    # example_use_of_gym_env()
    main()



# %%


# %%
