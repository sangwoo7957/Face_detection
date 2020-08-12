import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
from gaze_tracking import GazeTracking

# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 리턴
    return model

# 여러 사용자 학습
def trains():
    #faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model2 :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models    

#얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi_gray= gray[y:y+h, x:x+w]
            roi_color= img[y:y+h, x:x+w]
            eyes= eye_classifier.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh)in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),2)
            roi = cv2.resize(roi, (200,200))
        cv2.imshow("Face Detection", img)    
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

# 인식 시작
def run(models):    
    #카메라 열기 
    cap = cv2.VideoCapture(0)
    gaze = GazeTracking()
    cheat_cnt = 0
    right_cnt = 0
    left_cnt = 0
    up_cnt = 0
    down_cnt = 0
    name = ""
    
    while True:
        #카메라로 부터 사진 한장 읽기 
        ret, frame = cap.read()

        gaze.refresh(frame)
        # 얼굴 검출 시도 
        image, face = face_detector(frame)
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            #검출된 사진을 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    
            #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.         
            if min_score < 500:
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
                name = min_score_name
            cv2.putText(image,display_string,(90,170), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            #75 보다 크면 동일 인물로 간주해 Match!
            if confidence > 75:
                cv2.putText(image, "Match : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Detection', image)
                frame = gaze.annotated_frame()
                text = ""
                
                if gaze.is_blinking():
                    text = "Blinking"
                elif gaze.is_right():
                    text = "Looking right"
                    right_cnt += 1
                elif gaze.is_left():
                    text = "Looking left"
                    left_cnt += 1
                elif gaze.is_up():
                    text = "Looking up"
                    up_cnt += 1
                elif gaze.is_down():
                    text = "Looking down"
                    down_cnt += 1
                elif gaze.is_center():
                    text = "Looking center"

                cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()

                cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 135), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

                cv2.imshow("Face Detection", frame)

                if left_pupil is None and right_pupil is None:
                    cheat_cnt += 1
                elif right_cnt > 5:
                    cheat_cnt += 1
                    print('Too much looking right')
                elif left_cnt > 5:
                    cheat_cnt += 1
                    print('Too much looking left')
                elif up_cnt > 5:
                    cheat_cnt += 1
                    print('Too much looking up')
                elif down_cnt > 5:
                    cheat_cnt += 1
                    print('Too much looking down')
            else:
            #75 이하면 Unmatch 
                cv2.putText(image, "Unmatch", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Detection', image)
        except:
            #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Detection', image)
            pass
        
        if cheat_cnt > 100:
            print(name + " Cheating Probability is high")
            break
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    models = trains()
    run(models)
