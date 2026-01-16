import os
import sys
import contextlib
import pickle
import mediapipe as mp
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
DATA_DIR = './data'
EXPECTED_LENGTH = 84
MAX_WORKERS = None 

# --- 1. THE TRUE NUCLEAR SILENCER (OS-LEVEL) ---
@contextlib.contextmanager
def suppress_stderr():
    """
    Redirects the actual OS-level file descriptor (FD 2) to null.
    This catches C++ warnings that bypass sys.stderr.
    """
    # Open a null file
    with open(os.devnull, "w") as devnull:
        # Get the fileno for null
        null_fd = devnull.fileno()
        # Save the original stderr (FD 2) so we can restore it later
        saved_stderr_fd = os.dup(2)
        
        try:
            # Force FD 2 (stderr) to point to null
            os.dup2(null_fd, 2)
            yield
        finally:
            # Restore the original stderr
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    if iteration == total: 
        sys.stdout.write('\n')
    sys.stdout.flush()

def process_image(args):
    img_path, label = args
    
    # We apply the OS-level silencer right here
    # The warning happens when 'mp.solutions.hands' initializes the C++ graph
    with suppress_stderr():
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    return None
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x_.append(hand_landmarks.landmark[i].x)
                            y_.append(hand_landmarks.landmark[i].y)

                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                    if len(data_aux) < EXPECTED_LENGTH:
                        data_aux.extend([0.0] * (EXPECTED_LENGTH - len(data_aux)))

                    return (data_aux[:EXPECTED_LENGTH], label)
            except Exception:
                return None
    return None

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return

    # 1. Collect Tasks
    print("Scanning directories...")
    tasks = []
    dirs = [d for d in os.listdir(DATA_DIR) if d.isdigit()]
    
    for dir_ in dirs:
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue
        label = int(dir_)
        for img_name in os.listdir(dir_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append((os.path.join(dir_path, img_name), label))

    total_files = len(tasks)
    print(f"Found {total_files} images. Processing...")

    data = []
    labels = []
    
    print_progress_bar(0, total_files, prefix='Progress:', suffix='Complete', length=40)

    # 2. Execute Parallel Processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            
            if result is not None:
                features, label = result
                data.append(features)
                labels.append(label)
            
            print_progress_bar(i + 1, total_files, prefix='Progress:', suffix='Complete', length=40)

    # 3. Save Data
    print(f"Successfully processed {len(data)} valid samples.")
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print("Data processing complete. Saved to data.pickle")

if __name__ == '__main__':
    # Hiding warnings in the main process too, just in case
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()
