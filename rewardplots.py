import pandas as pd
pd.read_csv
import matplotlib.pyplot as plt
df1 = pd.read_csv('dataset/Rewards.csv', index_col=0)
df2 = pd.read_csv('dataset/RewardsPerActor.csv', index_col=0)
df3 = pd.read_csv('dataset/RewardDQN.csv', index_col=0)
df4 = pd.read_csv('dataset/RewardActorCritic.csv', index_col=0)
#df5 = pd.read_csv('dataset/RewardsonCall.csv', index_col=0)
#print(df1)
df1 = df1['Reward'].tolist()
df2 = df2['RewardActor'].tolist()
df3 = df3['RewardDQN'].tolist()
df4 = df4['RewardActorCritic'].tolist()
x = 4
df2 = [df2[i:i+x] for i in range(0, len(df2), x)]
Reward_du1 = []
Reward_du2 = []
Reward_du3 = []
NearRT = []
learner = []
for i in range(0, len(df2)):
    v = df2[i]
    for j in range(0, 4):
        if j == 0:
            Reward_du1.append(v[j])
        if j == 1:
            Reward_du2.append(v[j])
        if j == 2:
            Reward_du3.append(v[j])
        if j == 3:
            NearRT.append(v[j])
#print(Reward_du1)
#print(Reward_du2)
#print(Reward_du3)
plt.plot(Reward_du1, label="Reward per actor 1", marker='o')
plt.plot(Reward_du2, label="Reward per actor 2", marker='s')
plt.plot(Reward_du3, label="Reward per actor 3", marker='*')
plt.plot(NearRT, label="Reward per actor 4", marker='+')
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Number of episodes', fontsize=18)
plt.legend(fontsize=18)
plt.show()


plt.plot(df1[0:50], label="Ape-X based approach (our proposal) ", marker='o')
#plt.plot(df3[0:50], label="DQN based approach ", marker='*')
plt.plot(df4[0:50], label="Actor-Critic based approach", marker='s')
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Mean total reward', fontsize=18)
plt.xlabel('Number of episodes', fontsize=18)
plt.legend(fontsize=18)
plt.show()

#df5 = df5.cumsum()
#plt.figure()
#df5.plot()