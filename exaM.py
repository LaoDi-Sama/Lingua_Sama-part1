#题目：有一座塔7层，共有381盏灯，相邻两层的下一层灯盏数是上一层的2倍，求最顶层灯数

def check_floor(floor,all_lights):
    if floor == 1:
        return round(all_lights/2)
    return check_floor(floor-1,round(all_lights/2))

print(check_floor(3,7))# 1/2/4
print(check_floor(7,380))# 3/6/12/24/48/96/192 = 381

