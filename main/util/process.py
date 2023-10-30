import os,psutil,time
import signal
import subprocess
from gym_carla.setting import CARLA_PATH

operating_system='windows' if os.name=='nt' else 'linux'

def get_binary():
    return 'CarlaUE4.exe' if operating_system=='windows' else 'CarlaUE4.sh'

def get_exec_command():
    binary=get_binary()
    exec_command=binary if operating_system=='windows' else ('./'+binary+' -prefernvidia')

    return binary,exec_command

def kill_process():
    binary=get_binary()
    for process in psutil.process_iter():
        if process.name().lower().find(binary.split('.')[0].lower())!=-1:
            try:
                process.terminate()
            except:
                pass

    still_alive=[]
    for process in psutil.process_iter():
        if process.name().lower().find(binary.split('.')[0].lower())!=-1:
            still_alive.append(process)

    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

# Starts Carla simulator
def start_process():
    # Kill Carla processes if there are any and start simulator
    print('Starting Carla...')
    kill_process()
    subprocess.Popen(get_exec_command()[1],cwd=CARLA_PATH, shell=True)
    time.sleep(5)

def get_child_processes(logger, parent_pid):
    child_processes = set()
    try:
        # 使用ps命令获取父进程及其所有子进程的PID列表
        ps_command = f"ps -o pid,ppid -e | grep {parent_pid}"
        ps_output = os.popen(ps_command).read()
        ps_lines = ps_output.strip().split('\n')
        
        # 解析PID列表，包括父进程和子进程
        for line in ps_lines:
            parts = line.split()
            pid, ppid = int(parts[0]), int(parts[1])
            if ppid == parent_pid and psutil.pid_exists(pid):
                child_processes.add(pid)
        
    except Exception as e:
        logger.exception(f"获取子进程时发生错误：{str(e)}")
    
    return child_processes

def kill_process_and_children(logger, parent_pid):
    try:
        if psutil.pid_exists(parent_pid):
            # 获取子进程列表
            child_processes = get_child_processes(logger, parent_pid)
            
            # 杀死所有子进程
            for child_pid in child_processes:
                if operating_system == 'windows':
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(child_pid)])
                else:
                    name = psutil.Process(child_pid).name()
                    if name.find('CarlaUE4') != -1:
                        os.killpg(child_pid, signal.SIGKILL)
                    else:
                        os.kill(child_pid, signal.SIGKILL)
            
            # 杀死父进程
            os.kill(parent_pid, signal.SIGKILL)
        
        logger.info(f"进程 {parent_pid} 及其子进程已被终止。")
    except ProcessLookupError:
        logger.exception(f"进程 {parent_pid} 不存在。")
    except PermissionError:
        logger.exception(f"没有足够的权限来终止进程 {parent_pid} 及其子进程。")
    except Exception as e:
        logger.exception(f"发生错误：{str(e)}")