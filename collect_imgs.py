import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
capture_delay = 0.3  # I set this to 2 seconds so you have time to move

cap = cv2.VideoCapture(0)

while True:
    try:
        # 1. Select Class
        user_input = input("Enter the class number to record (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        j = int(user_input)
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {j}')

        # 2. Standby Phase
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Instruction text
            cv2.putText(frame, f'Class {j}: Press "Q" to start', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(25) == ord('q'):
                break

        # 3. Recording Phase
        counter = 0
        last_save_time = time.time()
        
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- DISPLAY INFO ON SCREEN ---
            
            # Show the Counter (How many captured so far)
            cv2.putText(frame, f"Captured: {counter} / {dataset_size}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # White text

            # Show Countdown to next photo (Optional but helpful)
            time_passed = time.time() - last_save_time
            time_left = max(0, capture_delay - time_passed)
            
            # Change color to RED if about to snap, GREEN if waiting
            timer_color = (0, 0, 255) if time_left < 0.5 else (0, 255, 0)
            
            cv2.putText(frame, f"Next photo: {time_left:.1f}s", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            # --- SAVE LOGIC ---
            if time_passed >= capture_delay:
                # Save the image
                save_path = os.path.join(class_dir, '{}.jpg'.format(counter))
                cv2.imwrite(save_path, frame)
                
                counter += 1
                last_save_time = time.time() # Reset timer
                print(f"Saved {save_path}")
        
        print(f"Finished recording class {j}.\n")

    except ValueError:
        print("Invalid input.")

cap.release()
cv2.destroyAllWindows()