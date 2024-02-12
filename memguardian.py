import subprocess
import psutil
import os
import re
import time
import signal
import sys
import ctypes
import pygetwindow as gw
import pyautogui

# Global flag to indicate whether a termination signal has been received
terminate_flag = False

# Signal handler for SIGINT (Ctrl+C)
def sigint_handler(signum, frame):
    global terminate_flag
    if terminate_flag:
        print("Force arresting...")
        sys.exit(0)
    print("Received Ctrl+C. Setting terminate flag...")
    terminate_flag = True

# Register the signal handler
signal.signal(signal.SIGINT, sigint_handler)

# Function to start breakout.py in a new shell and activate the environment
def start_breakout(upload_ep):
    command = 'start cmd /k "conda activate gymenv && C:/Users/franc/anaconda3/envs/gymenv/python.exe c:/Users/franc/Desktop/DQN-Breakout/breakout.py {}"'.format(upload_ep)
    process = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return process

# Function to find the maximum upload_ep from saved_models directory
def find_max_upload_ep():
    model_files = os.listdir('saved_models')
    model_files = [f for f in model_files if re.match(r'model_episode_\d+', f)]
    if not model_files:
        return 0
    return max(int(re.search(r'\d+', f).group()) for f in model_files)

# Function to gracefully terminate breakout.py
def terminate_breakout_pid(pid):
    try:
        process = psutil.Process(pid)
        process.send_signal(signal.CTRL_C_EVENT)
    except psutil.NoSuchProcess:
        pass

# Function to restart breakout.py with the maximum upload_ep
def restart_breakout():
    max_upload_ep = find_max_upload_ep()
    print("Restarting breakout.py with max_upload_ep:", max_upload_ep)
    return start_breakout(max_upload_ep)

def find_process_by_cmdline(target_cmdline):
    for process in psutil.process_iter(['pid', 'cmdline']):
        cmdline = process.info.get('cmdline', [])
        cmdline_str = ' '.join(cmdline) if cmdline else ""

        if target_cmdline == cmdline_str:
            return process.info['pid']
    return None

def find_shell_by_cmdline(target_cmdline):
    for process in psutil.process_iter(['pid', 'cmdline']):
        cmdline = process.info.get('cmdline', [])
        if cmdline == target_cmdline:
            return process.info['pid']

    return None

def get_memory_usage(pid):
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return (memory_info.rss+memory_info.vms) / (1024 ** 3)  # Convert to GB
    except psutil.NoSuchProcess:
        return None

while True:
    # Set the initial upload_ep
    upload_ep = find_max_upload_ep()
    breakout_process = None  # Initialize the variable to store the Popen object

    target_cmdline = 'C:/Users/franc/anaconda3/envs/gymenv/python.exe c:/Users/franc/Desktop/DQN-Breakout/breakout.py {}'.format(upload_ep)
    target_shell_cmdline = ['cmd', '/k', 'conda activate gymenv && C:/Users/franc/anaconda3/envs/gymenv/python.exe c:/Users/franc/Desktop/DQN-Breakout/breakout.py {}'.format(upload_ep)]
    window_title = 'C:\Windows\system32\cmd.exe - conda  activate gymenv  - C:/Users/franc/anaconda3/envs/gymenv/python.exe  c:/Users/franc/Desktop/DQN-Breakout/breakout.py'

    # Start breakout.py with the current upload_ep
    print("Starting breakout.py with upload_ep:", upload_ep)
    start_breakout(upload_ep)
    time.sleep(10)
    breakout_process_pid = find_process_by_cmdline(target_cmdline)
    shell_process_pid = find_shell_by_cmdline(target_shell_cmdline)
    print(breakout_process_pid)
    print(shell_process_pid)

    if breakout_process_pid:
        print(f"Found PID: {breakout_process_pid} for the target command line.")
        time.sleep(10)
        # Wait for breakout.py to finish or for termination signal
        while psutil.pid_exists(breakout_process_pid):
            if not terminate_flag:
                time.sleep(5)
                memory_usage = get_memory_usage(breakout_process_pid)

                if memory_usage is not None:
                    print("Process Memory Usage:", memory_usage, "GB")

                    # Check if memory usage is over a certain limit (GB)
                    if memory_usage > 36.5:
                        print("Memory excedeed threshold. Sending Ctrl+C to breakout.py...")
                        windows = gw.getAllTitles()

                        # Filter windows based on the title
                        matching_windows = [window for window in windows if window.startswith('C:\\Windows\\system32\\cmd.exe')]

                        console = gw.getWindowsWithTitle(matching_windows[0])[0]
                        console.minimize()
                        time.sleep(0.1)
                        console.restore()
                        time.sleep(0.1)
                        console.activate()

                        # Simulate Ctrl+C keypress
                        pyautogui.hotkey('ctrl', 'c')

                        while psutil.pid_exists(breakout_process_pid):
                            time.sleep(1)
                        
                        console.close()
                        breakout_process = None  # Initialize the variable to store the Popen object

                        restart_breakout()

                        breakout_process_pid = 0
                        shell_process_pid = 0
                        time.sleep(10)

            else:
                # Terminate breakout.py gracefully with Ctrl+C signal
                print("Sending Ctrl+C to breakout.py...")
                windows = gw.getAllTitles()

                # Filter windows based on the title
                matching_windows = [window for window in windows if window.startswith('C:\\Windows\\system32\\cmd.exe')]

                console = gw.getWindowsWithTitle(matching_windows[0])[0]
                console.minimize()
                time.sleep(0.1)
                console.restore()
                time.sleep(0.1)
                console.activate()

                # Simulate Ctrl+C keypress
                pyautogui.hotkey('ctrl', 'c')

                while psutil.pid_exists(breakout_process_pid):
                    time.sleep(1)
                
                console.close()
                sys.exit(0)
                
