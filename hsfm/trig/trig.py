import numpy as np


def calc_LR(x, y, URLON, URLAT, w, h, alpha, heading):
    delta_x = np.cos(np.radians(alpha)) * w
    delta_y = np.sin(np.radians(alpha)) * w
    
    if 0 < heading < 90:
        px = x + delta_y
        py = y - delta_x

    elif 90 < heading < 180:
        px = x - delta_x
        py = y - delta_y

    elif 180 < heading < 270:
        px = x - delta_y
        py = y + delta_x
            
    elif 270 < heading < 360:
        px = x + delta_x
        py = y + delta_y
    
    LRLON = 2*px - URLON
    LRLAT = 2*py - URLAT
    
    return LRLON, LRLAT
    
def calculate_corner(x,y,w,h,heading):
    
    angle = get_rectangle_angle_to_center(w, h)
    
    if heading == 0 or heading == 360:
        delta_x = w
        delta_y = h
        URLON = x + w
        URLAT = y + h
        LRLON = x + w
        LRLAT = y - h
        LLLON = x - w
        LLLAT = y - h
        ULLON = x - w
        ULLAT = y + h
        
    elif heading == 90:
        delta_x = w
        delta_y = h
        URLON = x + h
        URLAT = y - w
        LRLON = x - h
        LRLAT = y - w
        LLLON = x - h
        LLLAT = y + w
        ULLON = x + h
        ULLAT = y + w
        
    elif heading == 180:
        delta_x = w
        delta_y = h
        URLON = x - w
        URLAT = y - h
        LRLON = x - w
        LRLAT = y + h
        LLLON = x + w
        LLLAT = y + h
        ULLON = x + w
        ULLAT = y - h

    elif heading == 270:
        delta_x = w
        delta_y = h
        URLON = x - h
        URLAT = y + w
        LRLON = x + h
        LRLAT = y + w
        LLLON = x + h
        LLLAT = y - w
        ULLON = x - h
        ULLAT = y - w
    
    else: 
        if 0 < heading < 90:
            alpha = 90 - heading

            delta_x = np.cos(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)
            delta_y = np.sin(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)

            URLON = x + delta_y
            URLAT = y - delta_x
            LLLON = x - delta_y
            LLLAT = y + delta_x 

        elif 90 < heading < 180:
            alpha = 180 - heading

            delta_x = np.cos(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)
            delta_y = np.sin(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)

            URLON = x - delta_x
            URLAT = y - delta_y
            LLLON = x + delta_x
            LLLAT = y + delta_y 

        elif 180 < heading < 270:
            alpha = 270 - heading

            delta_x = np.cos(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)
            delta_y = np.sin(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)

            URLON = x - delta_y
            URLAT = y + delta_x
            LLLON = x + delta_y
            LLLAT = y - delta_x  

        elif 270 < heading < 360:
            alpha = 360 - heading

            delta_x = np.cos(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)
            delta_y = np.sin(np.radians(alpha + angle)) * np.sqrt(w**2 + h**2)

            URLON = x + delta_x
            URLAT = y + delta_y
            LLLON = x - delta_x
            LLLAT = y - delta_y



        LRLON, LRLAT = calc_LR(x, y, URLON, URLAT, w, h, alpha, heading)

        ULLON = 2*x - LRLON
        ULLAT = 2*y - LRLAT
        
    LR = (LRLON, LRLAT)
    UR = (URLON, URLAT)
    LL = (LLLON, LLLAT)
    UL = (ULLON, ULLAT)
    

    return UL, UR, LR, LL
    
def check_angle(point1,point2, point3):
    vector21 = np.array(point2) - np.array(point1)
    vector31 = np.array(point3) - np.array(point1)
    angle = np.dot(vector21, vector31)/(np.linalg.norm(vector21) * np.linalg.norm(vector31))
    angle = np.arccos(angle)    
    return np.degrees(angle)
       
def get_rectangle_angle_to_center(width, height):
    angle_to_center = np.degrees(np.arctan(height/width))
    return angle_to_center
    
