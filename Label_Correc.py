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
#In[]
# Motion function
def move_right(env):
    step(env,TR)
    step(env,MF)

def move_left(env):
    step(env,TL)
    step(env,MF)

def move_back(env):
    step(env,TR)
    step(env,TR)
    step(env,MF)

# def move_right(env):
#     step(env,TR)
#     step(env,MF)
#In[]
# When key is needed
def robot_motion(grid,env):
    l= env.agent_pos[1]
    b= env.agent_pos[0]

    right_grid_F=grid[l,b+1]
    left_grid_F=grid[l,b-1]
    top_grid_F=grid[l-1,b]
    bottom_grid_F=grid[l+1,b]

    a= (np.where(np.array([right_grid_F,left_grid_F,top_grid_F,bottom_grid_F]) < grid[l,b]))[0][0]
    dir=env.agent_dir
    print(a)
    print(dir)
    if(dir==0):
        if(a==0):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==1):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==2):
            print('TL-> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==3):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        # theta= np.pi/2 # 90
    elif(dir==1):
        if(a==0):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==1):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==2):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==3):
            print('MF')
            step(env,MF)
            return [MF]
        # theta=0
    elif(dir==2):
        if(a==0):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==1):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==2):
            print('TR-> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==3):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        # theta= -np.pi/2 #-90
    elif(dir==3):
        if(a==0):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==1):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==2):
            print('MF')
            step(env,MF)
            return [MF] 
        elif(a==3):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        # theta=np.pi


#In[]
# When key is not needed
def robot_2_Grid(grid,env):
    l= env.agent_pos[1]
    b= env.agent_pos[0]

    right_grid_F=grid[l,b+1]
    left_grid_F=grid[l,b-1]
    top_grid_F=grid[l-1,b]
    bottom_grid_F=grid[l+1,b]

    a= np.where(np.array([right_grid_F,left_grid_F,top_grid_F,bottom_grid_F]) < grid[l,b])[0] 
    dir=env.agent_dir
    print(a)
    print(dir)
    if(dir==0):
        if(a==0):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==1):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==2):
            print('TL-> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==3):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        # theta= np.pi/2 # 90
    elif(dir==1):
        if(a==0):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==1):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==2):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==3):
            print('MF')
            step(env,MF)
            return [MF]
        # theta=0
    elif(dir==2):
        if(a==0):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        elif(a==1):
            print('MF')
            step(env,MF)
            return [MF]
        elif(a==2):
            print('TR-> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==3):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        # theta= -np.pi/2 #-90
    elif(dir==3):
        if(a==0):
            print('TR -> MF')
            step(env,TR)
            step(env,MF)
            return [TR,MF]
        elif(a==1):
            print('TL -> MF')
            step(env,TL)
            step(env,MF)
            return [TL,MF]
        elif(a==2):
            print('MF')
            step(env,MF)
            return [MF] 
        elif(a==3):
            print('TR -> TR -> MF')
            step(env,TR)
            step(env,TR)
            step(env,MF)
            return [TR,TR,MF]
        # theta=np.pi


# In[]
def doorkey_problem(flag,c_CD,c_OD_1,c_OD_2,c_OD_3,goal,agentPos,keyPos,doorPos,env,info):
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
    goal=np.roll(goal,1)
    agentPos=np.roll(agentPos,1)
    keyPos=np.roll(keyPos,1)
    doorPos=np.roll(doorPos,1)

    if(flag==True):
        # print(c_CD)
        print('Key needed') # work with other 3 matrices here

        count1=c_OD_1[agentPos[0],agentPos[1]]-1
        print('count',count1)
        print('cost_grid',c_OD_1)

        seq=[]
        plot_env(env)
        while count1>=0:
            val=(robot_motion(c_OD_1,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            plot_env(env)
            print('count1',count1)
            count1=count1-1
        
        seq.pop(-1)
        # seq.pop(-1)
        seq.append(PK)
        step(env,PK)

        agentPos=env.agent_pos
        agentPos=np.roll(agentPos,1)

        # count2=c_OD_2[keyPos[0],keyPos[1]]-1
        count2=c_OD_2[agentPos[0],agentPos[1]]-1
        print(count2)
        print(c_OD_2)

        while count2>=0:
            val=(robot_motion(c_OD_2,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            plot_env(env)
            print('count2',count2)
            count2=count2-1
        
        # seq.pop(-1)
        seq.append(UD)
        step(env,UD)

        agentPos=env.agent_pos
        agentPos=np.roll(agentPos,1)

        # count3=c_OD_3[doorPos[0],doorPos[1]]
        count3=c_OD_3[agentPos[0],agentPos[1]]-1
        print(count3)
        print(c_OD_3)
        
        while count3>=0:
            val=(robot_motion(c_OD_3,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            plot_env(env)
            print('count3',count3)
            count3=count3-1
        # seq.pop(-1)
        
        optim_act_seq=seq
    else:
        print('Key not needed')
        count=c_CD[agentPos[0],agentPos[1]]
        print(count)
        seq=[]
        plot_env(env)
        while count>=0:
            val=(robot_2_Grid(c_CD,env))
            if(val):    
                for i in val:
                    seq.append(i)
            # elif(not val):
            #     seq.append(MF)
            plot_env(env)
            print(count)
            count=count-1
        print(c_CD)
        optim_act_seq=seq

        # print()
        
    # optim_act_seq=seq
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq

#In[]
def fill_4_cells(cost_grid,loc,val,grid_flag):
    l=loc[0]
    b=loc[1]
    # print(l,b)
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

    # a=min(np.max(cost_grid),cost_grid[l,b]+1)
    # print('a------------------------>>> ',a)
    return cost_grid,grid_flag,(cost_grid[l,b]+1)

#In[]
def label_Correction(env,agentPos,cost_grid,goal,grid_flag):

    c=0
    goal=np.roll(goal,1)
    agentPos=np.roll(agentPos,1)
    cost_grid,grid_flag,r = fill_4_cells(cost_grid, goal,0, grid_flag)
    # print('here')
    # print('Cost_grid')
    # print(cost_grid)
    # print('Grid_flag')
    # print(grid_flag)
    
    while c<=r:
        
        q=np.vstack((np.where(cost_grid==r)[0],np.where(cost_grid==r)[1]))
        # print(q)
        grid_flag[goal[0],goal[1]]=1
        # print(y)
        for i in range(len(q.T)):
            cost_grid,grid_flag,r= fill_4_cells(cost_grid, q[:,i].T,0, grid_flag)
        c=c+1

    # print('Cost_grid')
    # print(cost_grid)
    # print('Grid_flag')
    # print(grid_flag)

    
#In[]
def main():

    env_path = './envs/example-8x8.env'
    # env_path = './envs/doorkey-5x5-normal.env'
    # env_path = './envs/doorkey-6x6-direct.env' # gif saved
    # env_path = './envs/doorkey-6x6-normal.env' # PROBLEM
    # env_path = './envs/doorkey-6x6-shortcut.env' 
    # env_path = './envs/doorkey-8x8-direct.env' # gif saved
    # env_path = './envs/doorkey-8x8-normal.env'
    # env_path = './envs/doorkey-8x8-shortcut.env' 

    env, info = load_env(env_path) # load an environment

    goal=info['goal_pos']
    agentPos=info['init_agent_pos']
    keyPos=info['key_pos']
    doorPos=info['door_pos']
    
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))
    
    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0
    cost_grid_CD=cost_grid #######################################
    # print(cost_grid)

    #------------- When Door closed-----------
    # Cost without door
    label_Correction(env,agentPos,cost_grid,goal,grid_flag)

    if(cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ] ==0):
        cost_with_door_closed=math.inf
    else:
        cost_with_door_closed=cost_grid[info['init_agent_pos'][1],info['init_agent_pos'][0] ]
    
    #-------------When Door Open-------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    # ------------------init_pos to Key_pos
    label_Correction(env,agentPos,cost_grid,keyPos,grid_flag) 
    c1=cost_grid[agentPos[1],agentPos[0]]-1
    # print("C1= ",c1)
    cost_grid_OD_1=cost_grid ####################################

    # ------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # -----------------key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,keyPos,cost_grid,doorPos,grid_flag) 
    c2=cost_grid[keyPos[1],keyPos[0]]-1
    # print("C2= ",c2)
    cost_grid_OD_2=cost_grid #####################################
    #--------------------------
    world_grid= (gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T).astype(np.float32)
    index= np.where(world_grid!=1 )
    world_grid[index[0][:],index[1][:]]= math.inf
    world_grid[info['key_pos'][1],info['key_pos'][0]]=-2
    world_grid[info['goal_pos'][1],info['goal_pos'][0]]=0

    world_grid[np.where(world_grid==1)]=0
    grid_flag=np.zeros(np.shape(world_grid))

    cost_grid=world_grid
    cost_grid[np.where(cost_grid<=0)]=0

    # -----------------key to Door
    world_grid[info['door_pos'][1]][info['door_pos'][0]]=0 # Remove Door from Map
    label_Correction(env,doorPos,cost_grid,goal,grid_flag) 
    c3=cost_grid[doorPos[1],doorPos[0]]
    # print("C3= ",c3)
    cost_grid_OD_3=cost_grid #####################################
    
    

    cost_with_door_open=c1+c2+c3

    print('Cost with DOOR CLosed   ', cost_with_door_closed)
    print('Cost with DOOR Open  ', cost_with_door_open)

    if(cost_with_door_closed>cost_with_door_open):
        print('We need Key')
        flag=True
        # cost_grid_CD=0
    else:
        print('No key needed')
        flag=False
        # cost_grid_OD_1,cost_grid_OD_2,cost_grid_OD_3=0,0,0

    seq= doorkey_problem(flag,cost_grid_CD,cost_grid_OD_1,cost_grid_OD_2,cost_grid_OD_3,goal,agentPos,keyPos,doorPos,env,info)
    print(seq)
    plot_env(env)
    #----------------------------------------------------------------------------
    # seq = doorkey_problem(env) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0], path='./gif_new/example-8x8.gif') # draw a GIF & save


#In[]
if __name__ == '__main__':
    # example_use_of_gym_env()
    main()


# %%


# %%
