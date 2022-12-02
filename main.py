import sys
import threading
import time
import src.demo.naive_agent_groundtruth as na
import argparse
import json

class AgentThread(threading.Thread):
    def __init__(self, agent_configs, agent_ind):
        self.agent_ind = agent_ind
        self.agent_configs =  agent_configs
        threading.Thread.__init__(self)
    def run(self):
        agent = na.ClientNaiveAgent(self.agent_ind, self.agent_configs)
        agent.run()

def main(agent_configs, *args):
    for x in range(1):
        print('naive agent %s running'%str(x))
        agent = AgentThread( agent_configs, agent_ind = str(x))
        agent.start()
        time.sleep(5)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    default_agent_host = "127.0.0.1"
    default_agent_port = 2004
    default_observer_host = "127.0.0.1"
    default_observer_port = 2006

    try:
        #Wrapper of the communicating messages
        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
            default_agent_host = sc_json_config[0]["host"]
            default_agent_port = sc_json_config[0]["port"]

    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        print("server_client_config.json not found")


    try:
        with open('./src/client/server_observer_client_config.json', 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)
            default_observer_host = observer_sc_json_config[0]["host"]
            default_observer_port = observer_sc_json_config[0]["port"]
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        print("server_observer_client_config.json not found")

    parser = argparse.ArgumentParser(description = "default parser")

    parser.add_argument("-s", "--save_logs", type=str2bool, nargs='?',
                            const=True, default=True, required = False,
                            help="if you want to save the logs for all agents")
    parser.add_argument("-a", "--agent_host", default=default_agent_host, required = False,
                            help="host ip address")
    parser.add_argument("-p", "--agent_port" ,type=int, default=default_agent_port, required = False,
                            help="host port for action robot")
    parser.add_argument("-b", "--observer_host", default=default_observer_host, required = False,
                            help="host ip address")
    parser.add_argument("-o", "--observer_port", type=int, default=default_observer_port, required = False,
                            help="host port for observer agent")
    args = parser.parse_args()

    main(args)
