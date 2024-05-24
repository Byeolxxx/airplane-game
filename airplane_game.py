import pygame
import random
import torch
import torch.nn as nn

# 定义神经网络模型


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载模型
model = SimpleNN()
model.load_state_dict(torch.load('enemy_ai.pth'))
model.eval()

# 初始化Pygame
pygame.init()

# 屏幕尺寸
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 颜色定义
black = (0, 0, 0)
white = (255, 255, 255)

# 飞机类


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(white)
        self.rect = self.image.get_rect()
        self.rect.center = (screen_width / 2, screen_height - 50)
        self.speed = 5

    def update(self, keys_pressed):
        if keys_pressed[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys_pressed[pygame.K_RIGHT] and self.rect.right < screen_width:
            self.rect.x += self.speed

# 敌人类


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super(Enemy, self).__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen_width - self.rect.width)
        self.rect.y = random.randint(-100, -40)
        self.speed = random.randint(1, 3)

    def update(self):
        state = torch.tensor(
            [self.rect.x, self.rect.y, player.rect.x, player.rect.y], dtype=torch.float32)
        action = model(state).detach().numpy()
        self.rect.x += action[0]
        self.rect.y += action[1]

        # 如果敌人超出屏幕范围，则重新定位到屏幕内的随机位置
        if self.rect.top > screen_height or self.rect.left > screen_width or self.rect.right < 0:
            self.rect.x = random.randint(0, screen_width - self.rect.width)
            self.rect.y = random.randint(-100, -40)
            self.speed = random.randint(1, 3)


# 初始化pygame，创建敌人和玩家飞机
pygame.init()
all_sprites = pygame.sprite.Group()
enemies = pygame.sprite.Group()
num_enemies = 10
for i in range(num_enemies):
    enemy = Enemy()
    all_sprites.add(enemy)
    enemies.add(enemy)
player = Player()  # 创建玩家飞机
all_sprites.add(player)  # 将玩家飞机添加到all_sprites组中

# 游戏主循环
running = True
clock = pygame.time.Clock()
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取当前按下的键
    keys_pressed = pygame.key.get_pressed()

    # 更新玩家飞机的状态
    player.update(keys_pressed)

    # 更新敌人的状态
    enemies.update()

    # 绘制所有内容
    screen.fill((0, 0, 0))  # 填充背景色
    all_sprites.draw(screen)  # 绘制所有精灵
    pygame.display.flip()  # 更新整个显示
    clock.tick(30)  # 控制游戏速度

pygame.quit()  # 游戏结束，退出pygame
