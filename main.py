# RADUCU ALEXANDRU-FLORIAN - PROIECT INTELIGENTA ARTIFICIALA CarRacing-v2

# PROBLEME DE CARE M-AM INTALNIT SI LE-AM REZOLVAT:
# 1. CAND SUNT MAI MULTE SEGMENTE DE DRUM PREZENTE IN OBSERVATION, SE VA CALCULA CENTRUL ORICARUIA DINTRE SEGMENTE.
     #REZOLVARE: RESTRANGEREA ROI(REGIUNII DE INTERES) ASTFEL INCAT SA CUPRINDA DOAR CEL MAI APROPIAT DRUM DE MASINA
# 2. UNELE PATRATE CARE REPREZENTA IARBA ERAU CONSIDERATE DREPT SEGMENTE DE DRUM
     #REZOLVARE: 1. IMBUNATATIREA PID CONTROLLER-ULUI(ASTFEL INCAT MASINA SA NU MAI IASA DE PE DRUM PREA MULT, REDUCAND POSIBILITATEA DE INTALNIRE A ACELOR PATRATE)
     #           2. RESTRANGEREA PE LATIME A ROI

import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt # Necesara pentru DEBUGGING si pentru a vizualiza diversii pasi din image pre-processing
import keyboard # La apasarea tastei 'd' se deschide DEBUGGING MODE

# Configurez environmentu'
env = gym.make("CarRacing-v2", render_mode="human")
debug = 0 # Daca debug = 1, atunci se intra in DEBUGGING MODE
observation = env.reset()[0]
test = 3
previous_error=0.0 # Necesara in cazul in care eroarea curenta nu poate fi determinata
action = env.action_space.sample() # O actiune random
previous_road_center_x = 48 # Stocheaza coordonata X a centrului drumului din observation-ul precedent
previous_road_center_y = 48 # Stocheaza coordonata Y a centrului drumului din observation-ul precedent
previous_action = action # Memoreaza actiunea efectuata precedent
accelerate=0 # La apasarea tastei 'a' aceasta variabila este incrementata si adaugata la action. Cu cat acceleratia masinii e mai mare, cu atat e mai greu de pastrat masina pe drum
road_center_x = 48 # Stocheaza coordonata curenta X a centrului drumului
road_center_y = 48 # Stocheaza coordonata curenta Y a centrului drumului
car_position_x=48 # Pozitia masinii este intotdeauna 48

# Determin coordonatele (x,y) ale masinii
# P.S: Am creat aceasta functie inainte sa realizez ca environmentu' este cel care isi schimba coordonatele fata de masina. Masina ramane in centrul observation-ului, la coordonata X = 48
def detect_car_position(observation):
    # Preprocessing
    observation = observation[0:84, :, :]  # Decupez imaginea pentru a reduce nivelul computational
    observation = observation.mean(axis=2)  # Convert to grayscale
    observation = np.uint8(observation)
    # Extrag pozitia masinii
    edges = cv2.Canny(observation, 100, 200)  # Detectez marginile
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Caut contururile din imagine
    car_contour = None
    for contour in contours:
        if cv2.contourArea(contour) > 0:  # Elimin contururile mici
            if car_contour is None or cv2.contourArea(contour) > cv2.contourArea(car_contour):  # Caut cel mai mare contur detectat
                car_contour = contour
    if car_contour is not None:
        car_x, car_y, car_w, car_h = cv2.boundingRect(car_contour)  # Get bounding box
        car_center_x = car_x + car_w / 2
        car_center_y = car_y + car_h / 2
        if debug == 1:
            print("Pozitia masinii: ({}, {})".format(car_center_x, car_center_y))
        return car_center_x, car_center_y
    return None, None

# Determin coordonatele centrului drumului din fata masinii
def detect_road_center(observation):
    # Definesc regiunea de interes (ROI)
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    # Elimin tot ce nu este verde din imagine
    imask_green = mask_green>0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]

    height, width, _ = green.shape
    roi_top = int(height * 0.23)+5  # Definesc ROI
    roi_bottom = int(height * 0.5) + 20   # Definesc ROI
    roi = green[roi_top:roi_bottom, 23:73]

    # Preprocess
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 40,40, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determin marginile drumului
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ellipse)
    canny = cv2.Canny(closed, 50,150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug==1:
        # Afisez outputul fiecarui frame(PENTRU DEBUGGING)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(observation[0:84, : , :])
        ax[0, 0].set_title('Original')
        ax[0, 1].imshow(roi, cmap='gray')
        ax[0, 1].set_title('ROI')
        ax[1, 0].imshow(thresh, cmap='gray')
        ax[1, 0].set_title('Thresh')
        ax[1, 1].imshow(cv2.drawContours(observation.copy(), contours, -1, (0, 255, 0), 3))
        ax[1, 1].set_title('Contours')
        plt.show()

    nonZero = cv2.findNonZero(canny)
    try:
        middle = (nonZero[:, 0, 0].max() + nonZero[:, 0, 0].min()) / 2
        print(f'Mijlocul drumului: {middle}')
    except:
        print('Mijlocul drumului nu a putut fi determinat. Deschide debugging mode apasand tasta D')
    return middle+22, 1

# Determin urmatoarea actiune ce trebuie luata pentru mentinerea masinutei pe mijlocul drumului
def generate_next_action(road_center_x, road_center_y, previous_error=0.0):
    error = road_center_x - 48 # Cat de departe este masina de centrul drumului
    Kp = 0.02
    Ki = 0.03
    Kd = 0.15
    steering_angle = Kp * error + Ki * (error-previous_error) + Kd *(error-previous_error)
    print(f"Steering = {steering_angle} | Error: {error} | Previous Error: {previous_error}")
    action = [steering_angle, 0.03+accelerate, 0.01*(0.35*error)*0]
    print(f"Eroarea curenta: {error} | Actiunea: {action} | Centrul drumului: {road_center_x} , {road_center_y} ")
    return action, error

while True:
    try: # Debugging mode se porneste la apasarea tastei 'd'
        if keyboard.is_pressed('d'):
            if debug==0:
                debug=1
            else:
                debug=0
        if keyboard.is_pressed('a'):
            accelerate+=0.025
    except:
        print('Nu s-a putut deschide DEBUGGING MODE')
    test+=1

    try:
        road_center_x, road_center_y = detect_road_center(observation)
    except:
        previous_road_center_x = road_center_x
        previous_road_center_y = road_center_y
    if car_position_x != None and road_center_x != None:
       action, previous_error = generate_next_action(road_center_x, road_center_y, previous_error)
       previous_action = action
       observation, reward, done, terminated, info = env.step(action)
    else:
        action = env.action_space.sample()
        observation, reward, done, terminated, info = env.step(previous_action)
    if done:
        break


