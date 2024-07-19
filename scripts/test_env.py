from turtlebot_nav_env import Turtlebot_Nav_Env
import csv
import rospy


env = Turtlebot_Nav_Env()

# print("Resetting")
# env.reset()
# print("done")

NUM_EP = 1000
MAX_STEPS = 100000


with open("Rewards.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Rewards'])


for ep in range(NUM_EP):
    env.reset()
    print("Episode :", ep)
    for step in range(MAX_STEPS):
        obs, reward, done, info = env.step([0.5, 0])
        #print(reward)

        
