action_space = 65

def trans(t_action):
    if t_action == action_space:
        t_action = [1, 2, 3, 4]  # 1: keep allocation unchanged, 2:scale-up, 3: scale-down, 4 terminate
    elif t_action == action_space - 1:
        t_action = [3, 3, 2, 2]
    elif t_action == action_space - 2:
        t_action = [1, 2, 3, 4]
    elif t_action == action_space - 3:
        t_action = [1, 2, 3, 3]
    elif t_action == action_space - 4:
        t_action = [3, 2, 1, 2]
    elif t_action == action_space - 5:
        t_action = [1, 2, 1, 4]
    elif t_action == action_space - 6:
        t_action = [3, 2, 1, 3]
    elif t_action == action_space - 7:
        t_action = [3, 4, 1, 4]
    elif t_action == action_space - 8:
        t_action = [3, 1, 3, 3]
    elif t_action == action_space - 9:
        t_action = [3, 4, 1, 2]
    elif t_action == action_space - 10:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 11:
        t_action = [3, 2, 3, 4]
    elif t_action == action_space - 12:
        t_action = [2, 2, 3, 1]
    elif t_action == action_space - 13:
        t_action = [1, 2, 3, 4]
    elif t_action == action_space - 14:
        t_action = [3, 2, 1, 2]
    elif t_action == action_space - 15:
        t_action = [1, 2, 1, 3]
    elif t_action == action_space - 16:
        t_action = [4, 2, 1, 3]
    elif t_action == action_space - 17:
        t_action = [3, 2, 1, 1]
    elif t_action == action_space - 18:
        t_action = [1, 2, 1, 4]
    elif t_action == action_space - 19:
        t_action = [3, 2, 2, 4]
    elif t_action == action_space - 20:
        t_action = [1, 2, 3, 4]
    elif t_action == action_space - 21:
        t_action = [3, 2, 1, 1]
    elif t_action == action_space - 22:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 23:
        t_action = [3, 3, 3, 3]
    elif t_action == action_space - 24:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 25:
        t_action = [1, 1, 1, 1]
    elif t_action == action_space - 26:
        t_action = [2, 2, 2, 4]
    elif t_action == action_space - 27:
        t_action = [3, 2, 2, 2]
    elif t_action == action_space - 28:
        t_action = [3, 4, 4, 4]
    elif t_action == action_space - 29:
        t_action = [2, 2, 4, 4]
    elif t_action == action_space - 30:
        t_action = [1, 1, 1, 1]
    elif t_action == action_space - 31:
        t_action = [3, 4, 4, 4]
    elif t_action == action_space - 32:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 33:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 34:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 35:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 36:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 37:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 38:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 39:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 40:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 41:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 42:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 43:
        t_action = [3, 1, 1, 4]
    elif t_action == action_space - 44:
        t_action = [3, 3, 3, 4]
    elif t_action == action_space - 45:
        t_action = [3, 2, 3, 4]
    elif t_action == action_space - 46:
        t_action = [3, 2, 2, 3]
    elif t_action == action_space - 47:
        t_action = [3, 2, 1, 1]
    elif t_action == action_space - 48:
        t_action = [3, 3, 3, 4]
    elif t_action == action_space - 49:
        t_action = [3, 4, 1, 4]
    elif t_action == action_space - 50:
        t_action = [4, 4, 1, 4]
    elif t_action == action_space - 51:
        t_action = [1, 1, 2, 2]
    elif t_action == action_space - 52:
        t_action = [3, 2, 4, 4]
    elif t_action == action_space - 53:
        t_action = [2, 2, 1, 4]
    elif t_action == action_space - 54:
        t_action = [3, 2, 1, 4]
    elif t_action == action_space - 55:
        t_action = [1, 2, 3, 4]
    elif t_action == action_space - 56:
        t_action = [1, 2, 1, 4]
    elif t_action == action_space - 57:
        t_action = [2, 2, 4, 4]
    elif t_action == action_space - 58:
        t_action = [3, 4, 4, 4]
    elif t_action == action_space - 59:
        t_action = [3, 2, 3, 4]
    elif t_action == action_space - 60:
        t_action = [3, 2, 2, 4]
    elif t_action == action_space - 61:
        t_action = [1, 2, 4, 4]
    elif t_action == action_space - 62:
        t_action = [4, 2, 1, 4]
    elif t_action == action_space - 63:
        t_action = [3, 2, 3, 4]
    elif t_action == action_space - 64:
        t_action = [3, 2, 1, 3]
    else:
        t_action = [4, 3, 2, 1]
    return t_action