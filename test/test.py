matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

result = []
while matrix:
    result += matrix.pop(0)  # 取矩阵第一行并删除
    
    print(matrix)
    print(*matrix)
    print(list(zip(*matrix)))
    print(list(zip(*matrix))[::-1])
    print("------------------------")
    
    matrix = list(zip(*matrix))[::-1]  # 旋转矩阵