"""GOAL_STATE:
   0 : Personne dans la pièce
   1 : à definir
   2 : à définir
   3 : à définir
   4 : à définir
   6 : Le bébé est dans la pièce
   5 : Le parent est dans la pièce
   7 : Le bébé et le parent sont dans la pièce
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

def get_reward_matrix(G):
    global R
    Z = np.matrix([[-1,1,1,1,-1,-1,-1,-1],
                  [1,-1,-1,-1,1,-1,-1,1],
                  [1,-1,-1,-1,-1,1,-1,1],
                  [1,-1,-1,-1,1,1,-1,-1],
    	          [-1,1,-1,1,-1,-1,1,-1],
                  [-1,-1,1,1,-1,-1,1,-1],
                  [-1,-1,-1,-1,1,1,-1,1],
                  [-1,1,1,-1,-1,-1,1,-1]])
    def available_actions(state):
        current_state_row = Z[state,]
        av_act = np.where(current_state_row >= 0)[1]
        return av_act
    R = np.zeros(shape=[8,8], dtype=np.int)
    R[available_actions(G)[0],G] = 10
    R[available_actions(G)[1],G] = 10
    R[available_actions(G)[2],G] = 10
    R[G,G]= 10
    for i in range(0,G):
        R[i,i]=-10
    for j in range(G+1,8):
        R[j,j]=-10
    L0 = np.extract(available_actions(available_actions(G)[0]) != G,available_actions(available_actions(G)[0]))
    R[L0[0],available_actions(G)[0]] = 2
    R[L0[1],available_actions(G)[0]] = 2 
    L1 = np.extract(available_actions(available_actions(G)[1]) != G,available_actions(available_actions(G)[1]))
    R[L1[0],available_actions(G)[1]] = 2
    R[L1[1],available_actions(G)[1]] = 2
    L2 = np.extract(available_actions(available_actions(G)[2]) != G,available_actions(available_actions(G)[2]))
    R[L2[0],available_actions(G)[2]] = 2
    R[L2[1],available_actions(G)[2]] = 2
    L3 = available_actions(L0[0])[(available_actions(L0[0]) != available_actions(G)[2]) & (available_actions(L0[0]) != available_actions(G)[0])]
    R[L3,L0[0]] = 2
    R[L3,L0[1]] = 2
    R[L3,L1[0]] = 2
    R[R == 0] = -2
    R[0,4] = -10; R[0,5] = -10; R[0,6] = -10; R[0,7] = -10
    R[1,2] = -10; R[1,3] = -10; R[1,5] = -10; R[1,6] = -10
    R[2,1] = -10; R[2,3] = -10; R[2,4] = -10; R[2,6] = -10
    R[3,1] = -10; R[3,2] = -10; R[3,6] = -10; R[3,7] = -10
    R[4,0] = -10; R[4,2] = -10; R[4,5] = -10; R[4,7] = -10
    R[5,0] = -10; R[5,1] = -10; R[5,4] = -10; R[5,7] = -10
    R[6,0] = -10; R[6,1] = -10; R[6,2] = -10; R[6,3] = -10
    R[7,0] = -10; R[7,3] = -10; R[7,4] = -10; R[7,5] = -10
        
def get_poss_next_states(s, F, ns):
    poss_next_states = []
    for j in range(ns):
        if F[s,j] == 1: poss_next_states.append(j)
    return poss_next_states

def get_rnd_next_state(Q, s, F, ns, eps):
    al = random.random()
    poss_next_states = get_poss_next_states(s, F, ns)
    if al < eps :
        next_state = poss_next_states[np.random.randint(0,len(poss_next_states))]
    else :
        next_state = np.argmax(Q[s])
    return next_state

def train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
    eps = 0.5
    scores = []
    for i in range(0,max_epochs):
        curr_s = np.random.randint(0,ns)
        while(True):
            next_s = get_rnd_next_state(Q, curr_s, F, ns, eps)
            poss_next_next_states = get_poss_next_states(next_s, F, ns)
            max_Q = -9999.99
            for j in range(len(poss_next_next_states)):
                nn_s = poss_next_next_states[j]
                q = Q[next_s,nn_s]
                if q > max_Q:
                    max_Q = q
            Q[curr_s][next_s] = Q[curr_s][next_s] + lrn_rate * (R[curr_s][next_s] + gamma * max_Q - Q[curr_s][next_s])
            curr_s = next_s
            if curr_s == goal: break
        scores.append(np.max(Q))
        eps -= 0.0001
    plt.plot(scores)
    
def path_to_reach_goal_state(max_epochs,start, goal, Q):
    if max_epochs >= 5000 :
        epsilon = 0.1
    elif max_epochs >= 1000 :
        epsilon = 0.2
    else : 
        epsilon = 0.3
    al = random.random()
    curr = start
    print(str(curr) + "->", end="")
    while curr != goal:
        if al >= epsilon:
            next = np.argmax(Q[curr])
        else:
            next = np.random.randint(0,8)
        print(str(next) + "->", end="")
        curr = next
    print("done")

def main():
    F = np.ones(shape=[8,8], dtype=np.int)
    goal_state = int(input("Enter the goal_state: "))
    start_state = np.random.randint(0,8)
    get_reward_matrix(goal_state)
    number_of_states = 8
    gamma = 0.9
    lrn_rate = 0.5
    max_epochs = 5000
    if os.path.isfile("Q-tables"+str(goal_state)+".npy") == False :
        Q = np.zeros(shape=[8,8], dtype=np.float32)  
        train(F, R, Q, gamma, lrn_rate, goal_state, number_of_states, max_epochs)
        print("Using Q-table"+str(goal_state)+" to go from start_state: "+str(start_state)+ " to goal_state: "+str(goal_state))
        path_to_reach_goal_state(max_epochs,start_state, goal_state, Q)
        np.save("Q-tables/Q-tables"+str(goal_state),Q)
    else :
        Q = np.load("Q-tables"+str(goal_state)+".npy")
        print("Using existing Q-table"+str(goal_state)+" to go from start_state: "+str(start_state)+" to goal_state: "+str(goal_state))
        path_to_reach_goal_state(max_epochs,start_state, goal_state, Q)

if __name__ == '__main__':
    main()

