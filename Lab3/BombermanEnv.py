import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt


class Item(object):
    def __init__(self,pos):
        self.pos = pos

class Box(Item):
    def __init__(self,pos,breakable,broken,target):
        super(Box,self).__init__(pos)
        self.isBreakable = breakable
        self.isTarget = target
        self.isBroken = broken

    def get_state(self):
        return (self.pos, self.isBreakable, self.isTarget,self.isBroken)
    
class Enemy(Item):
    def __init__(self,pos,orientation,way,isAlive):
        super(Enemy,self).__init__(pos)
        self.orientation = orientation
        self.way = way
        self.isAlive = isAlive


    def get_state(self):
        return (self.pos, self.orientation, self.way, self.isAlive)
    


class Bomb(Item):
    def __init__(self,pos,timer):
        super(Bomb,self).__init__(pos)
        self.timer = timer

    def get_state(self):
        return (self.pos, self.timer)


class Explosion(Item):
    def __init__(self,pos):
        super(Explosion,self).__init__(pos)

    def get_state(self):
        return (self.pos)

def mult(x,y):
        if(x > y):
            multi = x/y
        else:
            multi = y/x
        return multi

def exist(list,pos):
    for i in range(len(list)):
        if(pos[0] == list[i].pos[0] and pos[1] == list[i].pos[1]):
            return True
    return False


class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, width, height, boxes, enemies_x, enemies_y, rompible_file, render_mode = None):
        super(BombermanEnv, self).__init__()

        self.width = width
        self.height = height
        if boxes < 1:
            print("El numero de cajas debe ser mayor que 1.")
            return 
        self.list_boxes = [] # lista para las cajas rompibles e irrompibles
        self.boxes = boxes # total de cajas rompibles
        self.list_boxes_breakable = [] # lista para las cajas rompibles 
        self.list_enemies = [] # lista que contendra a los enemigos
        self.enemies_x = enemies_x # total de enemigos horizontales
        self.enemies_y = enemies_y # total de enemigos verticales
        self.rompible_file = rompible_file # archivo de cajas irrompibles
        self._agent_location = np.array([0,0])
        self.active_bomb = 0
        self.observation = self._get_obs()
        m = mult(self.width, self.height)
        if(self.width > self.height):
            self.window_height = 670
            self.window_width = m * self.window_height
        else:
            self.window_width = 512
            self.window_height = m * self.window_width
            
        # Define la forma del espacio de observación (en este caso, una imagen binaria)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.observation),), dtype=np.float32)

        # Define el espacio de acción (puedes personalizar esto según tu entorno)
        self.action_space = spaces.Discrete(6)  # Ejemplo: acciones discretas 0, 1, 2, 3

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 0]),
        }
        self._action_to_names = {
            0: 'RIGHT',
            1: 'DOWN',
            2: 'LEFT',
            3: 'UP',
            4: 'BOMB',
            5: 'WAIT',
        }
        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define cualquier otro atributo específico de tu entorno
        self.window = None
        self.clock = None

        # Inicializa el estado inicial de tu entorno
        #self.state = np.zeros((width, height), dtype=np.float32)

    def distance(self, pos1, pos2):
        # buscamos la distancia entre el agente y un obstaculo
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_closest_obstacle(self, agent_position):
        """Encuentra la posición del obstáculo más cercano a la posición del agente."""
        closest_obstacle = None
        min_distance = float('inf')  # Inicializar con infinito para asegurar que cualquier distancia sea menor

        # Iterar sobre las cajas rompibles e irrompibles para encontrar la más cercana al agente
        for box in self.list_boxes + self.list_boxes_breakable:
            obstacle_position = box.pos
            d = self.distance(agent_position, obstacle_position)
            if d < min_distance:
                min_distance = d
                closest_obstacle = obstacle_position

        return closest_obstacle


    

    def _tile_is_free(self,direction):
        movement = self._agent_location + direction
        if (exist(self.list_boxes,movement)):
            return False
        return True

    def _get_obs(self):
        agent_position = self._agent_location
        closest_obstacle = self.find_closest_obstacle(agent_position)
        bomb_position = self.bomb.pos if self.active_bomb == 1 else np.array([0, 0])

        # Verificar si closest_obstacle no es None antes de aplanarlo
        if closest_obstacle is not None:
            closest_obstacle = closest_obstacle.flatten()
        else:
            # Si no hay obstáculo cercano, puedes proporcionar algún valor por defecto
            closest_obstacle = np.zeros_like(agent_position)

        # Aplana la información en un solo array
        flattened_observation = np.concatenate((
            agent_position.flatten(),
            closest_obstacle.flatten(),
            bomb_position.flatten()
        ))

        return np.array(flattened_observation)
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self._agent_location) - np.array(self._target_location), ord=1
            )
        }
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.list_boxes = [] # lista para las cajas rompibles e irrompibles
        self.list_boxes_breakable = [] # lista para las cajas rompibles
        self.list_enemies = [] # lista que contendra a los enemigos
        self.active_explosion = 0
        self.active_bomb = 0
        self.player_alive = True 
        self.bomb = Bomb(np.array([0,0]),0)
        self.explosion_radius = []
        # generamos las cajas irrompibles
        for i in range(self.width):
            for j in range(self.height):                        
                if ((i % 2 == 1) and (j % 2 == 1)):
                    self.list_boxes.append(Box(np.array([i,j]),False,False,False))
        
        # generamos las cajas rompibles
        if not self.rompible_file == '':
            rompible_info = []
            with open(self.rompible_file, "r") as file:
                for line in file:
                    rompible_info.append(line.strip())
            if not rompible_info == []:
                if len(rompible_info) == self.boxes:
                # Procesa la información y genera cajas irrompibles
                    rompible_coordinates = []
                    for line in rompible_info:
                        x, y = map(int, line.split(","))
                        rompible_coordinates.append((x, y))
                    for coordinates in rompible_coordinates:
                        self.list_boxes.append(Box(np.array(coordinates), True, False,False))
                        self.list_boxes_breakable.append(Box(np.array(coordinates), True, False,False))
        else:
            i=0
            while i < self.boxes:
                box_pos = np.array([self.np_random.integers(0,self.width-1,dtype=int),self.np_random.integers(0,self.height-1,dtype=int)])
                for j in range(len(self.list_boxes)):
                    if not (exist(self.list_boxes,box_pos)):
                        self.list_boxes.append(Box(box_pos,True,False,False))
                        self.list_boxes_breakable.append(Box(box_pos,True,False,False))
                        i+=1

        '''for i in range(len(self.list_boxes_breakable)):
            print(self.list_boxes_breakable[i].pos)'''
        # Choose the agent's location uniformly at random
        '''aux = True
        while (aux):
            x = self.np_random.integers(0, self.width, dtype=int)
            y = self.np_random.integers(0, self.height, dtype=int)
            if not (exist(self.list_boxes,np.array([x,y]))):
                aux = False
                self._agent_location = np.array([x,y])
            '''
        self._agent_location = np.array([2,2])

        # Choose target location between breakable boxes
        target_box = self.list_boxes_breakable[self.np_random.integers(0,len(self.list_boxes_breakable),dtype=int)]
        for i in range(len(self.list_boxes)):
            if (np.array_equal(target_box.pos,self.list_boxes[i].pos)):
                self.list_boxes[i].isTarget = True
                self._target_location = target_box.pos
                self._target_index = i


        # Choose random position for the enemies in horizontal axis
        '''if (self.enemies_x > 0):
            i = 0
            while i < self.enemies_x:
                m = self.np_random.integers(0, self.width, dtype=int)
                n = self.np_random.integers(0, self.height, dtype=int)
                if not (exist(self.list_boxes,np.array([m,n])) or np.array_equal(self._agent_location,np.array([m,n])) or exist(self.list_enemies,np.array([m,n]))):
                    w = self.np_random.integers(0, 2, dtype=int) # 0: Up, 1: Down
                    self.list_enemies.append(Enemy(np.array([m,n]),0,w,True)) # 0 en orientation es horizontal
                    i+=1
        # Choose random position for the enemies in vertical axis
        if (self.enemies_y > 0):
            i = 0
            while i < self.enemies_y:
                o = self.np_random.integers(0, self.width, dtype=int)
                p = self.np_random.integers(0, self.height, dtype=int)
                if not (exist(self.list_boxes,np.array([o,p])) or np.array_equal(self._agent_location,np.array([o,p])) or exist(self.list_enemies,np.array([o,p]))):
                    w = self.np_random.integers(0, 2, dtype=int) # 0: Left, 1: Right
                    self.list_enemies.append(Enemy(np.array([o,p]),1,w,True)) # 1 en orientation es vertical
                    i+=1'''
        enemigo_fijo = Enemy(np.array([4,4]),0,0,True)
        self.list_enemies.append(enemigo_fijo)
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info



    def step(self, action):
        reward = 0
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        #print(self._action_to_names[action])
        # We use `np.clip` to make sure we don't leave the grid
        x,y = self._agent_location + direction
        #print(x,y)

        #-------------------------------------Bomb Explosion-------------------------------------------#
        if (self.active_bomb == 1):
            # hay una bomba activa, por lo que se debe verificar si exploto
            if (self.bomb.timer == 0):
                # calculamos el radio de explosion que es un bloque mas que el radio de la bomba
                self.explosion_radius = []
                for i in range(4):
                    self.explosion_radius.append(Explosion(np.array(self.bomb.pos + self._action_to_direction[i])))
                self.explosion_radius.append(Explosion(self.bomb.pos))
                self.active_explosion = 1
                # verificamos si el agente esta en el radio de explosion
                if (exist(self.explosion_radius,self._agent_location)):
                    self.player_alive = False
                    observation = self._get_obs()
                    info = self._get_info()
                    reward = -10
                    terminated = 0
                    return observation, reward, terminated, True, info
                
                # verificamos si las cajas rompibles estan en el radio de explosion
                for i in range(len(self.list_boxes)):
                    if (self.list_boxes[i].isBreakable):
                        if (exist(self.explosion_radius,self.list_boxes[i].pos)):
                            self.list_boxes[i].isBroken = True
                            reward += 7
                            if (self.list_boxes[i].isTarget):
                                reward += 20

                # verificamos si los enemigos estan en el radio de explosion
                for i in range(len(self.list_enemies)):
                    if (exist(self.explosion_radius,self.list_enemies[i].pos)):
                        self.list_enemies[i].isAlive = False
                        # faltaria cambiar la recompenza obtenida por matar un enemigo
                        reward += 5
                self._render_frame() 
                self.active_bomb = 0
                self.explosion_radius = []
            else:
                self.bomb.timer -= 1
                            

                

        #-------------------------------------Agent movement-------------------------------------------#
        if (action == 2 or action == 0) and self._tile_is_free(direction):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.width - 1
            )
        elif (action == 1 or action == 3) and self._tile_is_free(direction):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.height - 1
            )

        #-------------------------------------Hit an enemy-------------------------------------------#
        for i in range(len(self.list_enemies)):
            if self.list_enemies[i].isAlive:
                if (np.array_equal(self._agent_location,self.list_enemies[i].pos)):
                    self.player_alive = False
                    observation = self._get_obs()
                    info = self._get_info()
                    reward = -10
                    terminated = 0
                    return observation, reward, terminated, True, info

        
        #-------------------------------------Enemy Movement-------------------------------------------#
        for i in range(len(self.list_enemies)):
            if (self.list_enemies[i].isAlive): # verificamos que este vivo
                # Separamos el movimiento entre vertical y horizontal
                if self.list_enemies[i].orientation == 0: #movimiento horizontal
                    if self.list_enemies[i].way == 0: #movimiento a la izquierda
                        # verificamos que no choque con el limite
                        if (self.list_enemies[i].pos[0] == 0) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[2]) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[2]))): # llego al limite de la izquierda, entonces lo cambiamos de sentido
                            self.list_enemies[i].way = 1
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4] # lo dejamos quieto
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[0] #lo movemos a la derecha
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[2] # lo movemos a la izquierda

                    else: #movimiento a la derecha
                        # verificamos que no choque con el limite
                        if (self.list_enemies[i].pos[0] == self.width-1) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[0]))):
                            self.list_enemies[i].way = 0
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4] # lo dejamos quieto
                            else:   
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[2] # lo movemos a la izquierda
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[0] # lo movemos a la derecha
                
                
                else:
                    # movimiento vertical
                    if self.list_enemies[i].way == 0: # movimiento hacia arriba
                        # se verifica que no choque con nada
                        if ((self.list_enemies[i].pos[1] == 0) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[3])) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[3]))):
                            self.list_enemies[i].way = 1
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[1]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4]
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[1] # lo movemos hacia abajo
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[3] # lo movemos hacia arriba
                    else: # movimiento hacia abajo
                        # se verifica que no choque con nada
                        if ((self.list_enemies[i].pos[1] == self.height-1) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[1])) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[1]))):
                            self.list_enemies[i].way = 0
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[3]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4]
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[3] # lo movemos hacia arriba
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[1] # lo movemos hacia abajo


        #-------------------------------------Bomb Placement-------------------------------------------#
        if (action == 4) and (self.active_bomb == 0):
            # no hay una bomba activa, por lo que se puede poner una
            self.active_bomb = 1
            self.bomb = Bomb(self._agent_location,6)
            reward = 1
            # si el target esta visible, debe dejar de ganar recomensa por colocar bombas
            if (self.list_boxes[self._target_index].isBroken):
                reward = 0

        
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        if (np.array_equal(self._agent_location,self._target_location)):
            reward = 100
            print('                 Ganaste\n')
            terminated = 1
            return observation, reward, terminated, True, info

        # verificamos si la salida esta expuesta, de estarlo, la recompensa debiese tornarse negativa para que 
        # agente no rompa todas las cajas antes de salirse
        '''if (self.list_boxes[self._target_index].isBroken):
            reward = -5'''
        return observation, reward, terminated, False, info




    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Mi Juego Genial")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        if(self.width > self.height):
            pix_square_size = (
                self.window_width / self.width
        )  # The size of a single grid square in pixels
        else:
            pix_square_size = (
                self.window_height / self.height
            )

        # ---------------------------Now we draw the agent -------------------------
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # ------------------------------------Now we draw the boxes------------------------------------
        for i in range(len(self.list_boxes)):
            if (self.list_boxes[i].isBreakable):
                if(self.list_boxes[i].isTarget and self.list_boxes[i].isBroken): # target visible
                    pygame.draw.rect(
                        canvas,
                        (52, 118, 32),
                        pygame.Rect(
                            (self.list_boxes[i].pos * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif not (self.list_boxes[i].isBroken): #cajas todavia visibles
                    pygame.draw.rect(
                    canvas,
                    (157, 124, 63),
                    pygame.Rect(
                        (self.list_boxes[i].pos * pix_square_size),
                        (pix_square_size, pix_square_size),
                    ),
                )
            else: # undestructible boxes
                pygame.draw.rect(
                    canvas,
                    (88, 82, 72),
                    pygame.Rect(
                        (self.list_boxes[i].pos * pix_square_size),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # ------------------------------------Now we draw the enemies------------------------------------
        for i in range(len(self.list_enemies)):
            if self.list_enemies[i].isAlive:
                pygame.draw.circle(
                canvas,
                (184, 22, 37),
                (self.list_enemies[i].pos + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # ------------------------------------Now we draw the bombs------------------------------------
        if (self.active_bomb == 1):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                (self.bomb.pos + 0.4) * pix_square_size,
                pix_square_size / 3,
            )

        # ------------------------------------Now we draw the explosion------------------------------------
        if (self.active_explosion == 1):
            for i in range(len(self.explosion_radius)):
                pygame.draw.circle(
                    canvas,
                    (206, 125, 9),
                    (self.explosion_radius[i].pos + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

        # Finally, add some gridlines
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=3,
            )
        
        for y in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * y, 0),
                (pix_square_size * y, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )



