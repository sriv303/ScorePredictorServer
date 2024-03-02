ball_list = ["" for k in range(0, 300)]
for i in range(0, 50):
    for j in range(1, 7):
        current_ball = i + 0.1 * j
        ball_list[i*6 + j-1] = round(current_ball, 1)

print(ball_list)
