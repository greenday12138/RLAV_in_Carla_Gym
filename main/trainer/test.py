import gym
import logging
from gym_carla.single_lane.carla_env import CarlaEnv
from gym_carla.single_lane.settings import ARGS
from temp import Temp

def main():
    args=ARGS.parse_args()

    log_level=logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    #env=gym.make('CarlaEnv-v0')
    env=CarlaEnv(args)
    #e=Temp(args)
    
    done=False
    truncated=False
    with open('./out/state_test.txt','w') as f:
        pass
    state=env.reset()

    try:
        while not done and not truncated:
            action=[[2.0,0.0]]
            next_state,reward,truncated,done,info=env.step(action)
            
            with open('./out/state_test.txt','a') as f:
                f.write(str(state['waypoints'])+'\n')
                f.write(str(state['ego_vehicle']) + '\n')
                f.write(str(state['vehicle_front'])+'\n')
                f.write(str(next_state['waypoints']) + '\n')
                f.write(str(next_state['ego_vehicle']) + '\n')
                f.write(str(next_state['vehicle_front']) + '\n')
                # for loc in state['waypoints']:
                #     f.write(str(loc)+'\n')
                f.write(str(reward)+'\t'+str(info)+'\n')
                f.write('\n')
            state=next_state
            #e.step()
    except KeyboardInterrupt:
        pass
    finally:
        env.__del__()
        logging.info('\nDone.')

if __name__=='__main__':
    main()