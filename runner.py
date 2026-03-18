import subprocess
import sys
import os

def run_app():
    # MediaPipe 임포트 테스트 및 환경 확인용 스크립트 실행 없이 바로 본체 실행
    # (이미 standalone-test로 'import mediapipe.solutions.hands' 등이 되는 것을 확인했으므로)
    
    app_path = os.path.join(os.getcwd(), 'app.py')
    print(f"Launching application: {app_path}")
    
    try:
        # 현재 파이썬 실행기로 app.py를 실행
        process = subprocess.Popen([sys.executable, app_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    run_app()
