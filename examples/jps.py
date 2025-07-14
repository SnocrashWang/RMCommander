import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import time

def is_valid(grid, pos):
    x, y = pos
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0

def has_forced_neighbor(grid, pos, dir):
    x, y = pos
    dx, dy = dir
    
    # 对角线移动时的强制邻居检查
    if dx != 0 and dy != 0:
        # 检查对角线方向的强制邻居
        if (is_valid(grid, (x - dx, y)) and not is_valid(grid, (x - dx, y + dy))):
            return True
        if (is_valid(grid, (x, y - dy)) and not is_valid(grid, (x + dx, y - dy))):
            return True
        return False
    
    # 水平移动时的强制邻居检查
    if dx == 0:
        if dy > 0:  # 向右移动
            if (is_valid(grid, (x + 1, y)) and not is_valid(grid, (x + 1, y + dy))) or \
               (is_valid(grid, (x - 1, y)) and not is_valid(grid, (x - 1, y + dy))):
                return True
        else:  # 向左移动
            if (is_valid(grid, (x + 1, y)) and not is_valid(grid, (x + 1, y + dy))) or \
               (is_valid(grid, (x - 1, y)) and not is_valid(grid, (x - 1, y + dy))):
                return True
    
    # 垂直移动时的强制邻居检查
    else:
        if dx > 0:  # 向下移动
            if (is_valid(grid, (x, y + 1)) and not is_valid(grid, (x + dx, y + 1))) or \
               (is_valid(grid, (x, y - 1)) and not is_valid(grid, (x + dx, y - 1))):
                return True
        else:  # 向上移动
            if (is_valid(grid, (x, y + 1)) and not is_valid(grid, (x + dx, y + 1))) or \
               (is_valid(grid, (x, y - 1)) and not is_valid(grid, (x + dx, y - 1))):
                return True
    
    return False

def jump(grid, current, dir, goal):
    dx, dy = dir
    x, y = current
    nx, ny = x + dx, y + dy
    
    # 如果下一个位置不可通行，直接返回None
    if not is_valid(grid, (nx, ny)):
        return None
    
    # 如果到达目标，返回目标位置
    if (nx, ny) == goal:
        return (nx, ny)
    
    # 检查是否有强制邻居
    if has_forced_neighbor(grid, (nx, ny), dir):
        return (nx, ny)
    
    # 对角线移动时需要检查直线方向
    if dx != 0 and dy != 0:
        # 检查水平方向
        if jump(grid, (nx, ny), (dx, 0), goal) is not None:
            return (nx, ny)
        # 检查垂直方向
        if jump(grid, (nx, ny), (0, dy), goal) is not None:
            return (nx, ny)
    
    # 继续沿原方向跳跃
    return jump(grid, (nx, ny), dir, goal)

def get_successors(grid, node, goal):
    successors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
                
            jump_point = jump(grid, node, (dx, dy), goal)
            if jump_point:
                successors.append(jump_point)
    
    return successors

def jps_search(grid, start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        _, current = open_set.get()
        
        if current == goal:
            return reconstruct_path(came_from, goal)
        
        successors = get_successors(grid, current, goal)
        
        for neighbor in successors:
            # 计算从当前节点到邻居的实际距离
            tentative_g = g_score[current] + distance(current, neighbor)
            
            # 如果找到更短的路径
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                
                # 如果邻居不在开放列表中，添加它
                open_set.put((f_score[neighbor], neighbor))
    
    return []  # 没有找到路径

def heuristic(a, b):
    # 使用对角线距离作为启发式函数
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return 10 * (dx + dy) + (14 - 2 * 10) * min(dx, dy)

def distance(a, b):
    # 计算实际距离
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    
    if dx == 0 or dy == 0:
        # 直线移动
        return 10 * max(dx, dy)
    else:
        # 对角线移动
        return 14 * min(dx, dy) + 10 * abs(dx - dy)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 反转路径

def visualize(grid, path, start, goal):
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='binary', origin='upper')
    
    # 绘制起点和终点
    plt.scatter(start[1], start[0], c='green', s=200, marker='o', label='Start')
    plt.scatter(goal[1], goal[0], c='red', s=200, marker='x', label='Goal')
    
    # 绘制路径
    if path:
        path_x, path_y = zip(*[(p[0], p[1]) for p in path])
        plt.plot(path_y, path_x, c='blue', linewidth=3, label='Path')
        
        # 绘制路径点
        plt.scatter(path_y, path_x, c='cyan', s=50, marker='o')
    
    # 添加网格和标签
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    plt.title('Jump Point Search Pathfinding')
    plt.legend()
    plt.show()

# 创建测试地图 - 确保起点和终点之间有路径
grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (7, 7)

print("Starting JPS search...")
start_time = time.time()
path = jps_search(grid, start, goal)
elapsed_time = time.time() - start_time

if path:
    print(f"Path found in {elapsed_time:.4f} seconds")
    print(f"Path length: {len(path)} steps")
    print(f"Path: {path[:5]}...{path[-5:]}")
    visualize(grid, path, start, goal)
else:
    print("No path found!")