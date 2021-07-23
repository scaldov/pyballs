#!/usr/bin/python3
import math
import time
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import argparse


class Ball:
    pass


class Wall:
    pass


class View:
    def __init__(self, **kwargs):
        self.linear = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.translate = np.array([0.0, 0.0])

    def set(self, w, h, xres, yres):
        if w * yres > h * xres:
            s = h / yres
        else:
            s = w / xres
        self.translate = np.array([w / 2,  h / 2])
        self.linear = np.array([[s / 2, 0.], [0., -s / 2]])

    def transform(self, v):
        return self.linear.dot(v) + self.translate

    def transformS(self, v):
        return self.linear.dot(v)

class Obj:
    def __init__(self, **kwargs):
        self.collision_checked = False
        self.poly = list()
        self.center = np.array([0.0, 0.0])
        self.vel = np.array([0., 0.])
        self.accel = np.array([0.0, 0.0])
        self.static = False
        self.mass = 1
        self.__dict__.update(kwargs)
        self.center = np.array(self.center)
        self.vel = np.array(self.vel)
        self.accel = np.array(self.accel)

    def calc_vel(self, dt):
        self.vel += self.accel * dt

    def calc_coord(self, dt):
        self.center += self.vel * dt

    def draw(self, painter):
        return


class Ball(Obj):
    def __init__(self, **kwargs):
        self.radius = 1.
        self.accel = np.array([0.0, -2.0])
        Obj.__init__(self, **kwargs)
        # print(kwargs)
        # print(kwargs['radius'])
        self.mass *= self.radius**2

    def draw(self, painter, view):
        r = view.linear.dot(np.array([self.radius, self.radius]))
        v = view.transform(self.center)
        painter.drawEllipse(int(v[0] - r[0]), int(v[1] - r[1]), r[0] * 2, r[1] * 2)

class Wall(Obj):
    def __init__(self, **kwargs):
        self.norm = np.array([0.0, 1.0])
        Obj.__init__(self, **kwargs)
        self.norm = np.array(self.norm)
        self.norm = self.norm / np.linalg.norm(self.norm)
        self.mass = 1e100
        self.static = True
        self.accel = np.array([0.0, 0.0])

    def draw(self, painter, view):
        e = np.array([self.norm[1], - self.norm[0]])
        v1 = self.center - self.len * 0.5 * e
        v2 = self.center + self.len * 0.5 * e
        v1 = view.transform(v1)
        v2 = view.transform(v2)
        painter.drawLine(v1[0], v1[1], v2[0], v2[1])


class Physics:
    def __init__(self, **kwargs):
        self.fmap = dict()
        self.fmap[Ball().__class__.__name__] = dict()
        self.fmap[Ball().__class__.__name__][Ball().__class__.__name__] = self.ball_ball
        self.fmap[Ball().__class__.__name__][Wall().__class__.__name__] = self.ball_wall
        self.__dict__.update(kwargs)

    def collide(self, obj1, obj2):
        # print(obj1.__class__.__name__, obj2.__class__.__name__)
        if self.fmap.get(obj1.__class__.__name__) is None:
            obj1, obj2 = obj2, obj1
            if self.fmap.get(obj1.__class__.__name__) is None:
                # print('no function @', obj1.__class__.__name__, obj2.__class__.__name__)
                return
        if self.fmap[obj1.__class__.__name__].get(obj2.__class__.__name__) is None:
            # print('no function @', obj1.__class__.__name__, obj2.__class__.__name__)
            return
        else:
            self.fmap[obj1.__class__.__name__][obj2.__class__.__name__](obj1, obj2)

    def ball_ball(self, obj1: Ball, obj2: Ball):
        if np.linalg.norm(obj1.center - obj2.center) > (obj1.radius + obj2.radius):
            return
        else:
            v1 = obj1.vel
            v2 = obj2.vel
            x1 = obj1.center
            x2 = obj2.center
            dv = v2 - v1
            dx = x2 - x1
            m1 = obj1.mass
            m2 = obj2.mass
            ms = m1 + m2
            du = dx.dot(dv) * dx / dx.dot(dx)
            u1 = v1 + 2. * m2 / ms * du
            u2 = v2 - 2. * m1 / ms * du
            obj1.vel = u1
            obj2.vel = u2
            # calc time to repell balls from each other
            dv = u2 - u1
            dv2 = dv.dot(dv)
            if dv2 < 1e-12: # too small rel velocity
                return
            dx2 = dx.dot(dx)
            dxdv = dx.dot(dv)
            r1 = obj1.radius
            r2 = obj2.radius
            det = dxdv**2 - dv2 * (dx2 - (r1 + r2)**2)
            if det < 0.: # too small rel velocity
                return
            dt = - (dxdv - np.sqrt(det)) / dv2
            # update centers
            obj1.center = x1 + dt * u1
            obj2.center = x2 + dt * u2

    def ball_wall(self, ball: Ball, wall: Wall):
        e = np.array([wall.norm[1], - wall.norm[0]])
        d = ball.center - wall.center
        if np.abs(d.dot(e)) < wall.len * 0.5 and 0 < d.dot(wall.norm) < ball.radius:
            # inside envelope rectangle
            ball.vel = ball.vel - 2.0 * wall.norm * ball.vel.dot(wall.norm)
            ball.center = ball.center - wall.norm * ((ball.center - wall.center).dot(wall.norm) - ball.radius)
        else: # maybe around corners
            p1 = wall.center + e * wall.len * 0.5
            p2 = wall.center - e * wall.len * 0.5
            d1 = ball.center - p1
            d2 = ball.center - p2
            r2 = ball.radius ** 2
            if d1.dot(d1) < r2:
                # corner 1
                n = d1 / np.linalg.norm(d1)
                ball.vel = ball.vel - 2.0 * n * ball.vel.dot(n)
                ball.center = ball.center - n * ((ball.center - p1).dot(n) - ball.radius)
            elif d2.dot(d2) < r2:
                #corner 2
                n = d2 / np.linalg.norm(d2)
                ball.vel = ball.vel - 2.0 * n * ball.vel.dot(n)
                ball.center = ball.center - n * ((ball.center - p2).dot(n) - ball.radius)
                


class Scene(QWidget):
    def __init__(self, xres, yres):
        super().__init__()
        self.xres = xres
        self.yres = yres
        # QWidget.__init__(self, **kwargs)
        self.objects = list()
        self.timer = QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.setInterval(int(1000 / 60))
        self.timer.start()
        self.ball = (0, 0)
        self.iframe = 0

    def timerEvent(self):
        phys = Physics()
        for o in self.objects:
            o.collision_checked = False
        for o in self.objects:
            o.calc_vel(0.01)
        for o in self.objects:
            o.calc_coord(0.01)
        for j in range(0, len(self.objects)):
            for i in range(j + 1, len(self.objects)):
                o1 = self.objects[j]
                o2 = self.objects[i]
                if o2 is not o1:
                    phys.collide(o1, o2)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        view = View()
        view.set(painter.device().width(), painter.device().height(), self.xres, self.yres)
        for o in self.objects:
            o.draw(painter, view)
        painter.end()

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xres', help='x resolution', required = False, default = 1.)
parser.add_argument('-y', '--yres', help='x resolution', required = False, default = 1.)
args, unknown_args = parser.parse_known_args()

app = QApplication([])
scene = Scene(float(args.xres), float(args.yres))
scene.objects.append(Ball(radius=0.1, vel=[0.2, 1.0], center=[0.2, 0.5]))
scene.objects.append(Ball(radius=0.1, vel=[0.3, 1.0], center=[0.4, 0.5]))
scene.objects.append(Ball(radius=0.1, vel=[0.22, 0.7], center=[-0.2, 0.5]))
scene.objects.append(Ball(radius=0.1, vel=[0.17, 0.7], center=[-0.6, 0.8]))
scene.objects.append(Ball(radius=0.1, vel=[0.17, 0.7], center=[-0.0, 0.2]))
scene.objects.append(Ball(radius=0.2, vel=[0.17, 0.7], center=[-0.0, -0.1]))
# scene.objects.append(Wall(center=[0.0, -0.9], norm = [0.0, 1.], len = 1))
# scene.objects.append(Wall(center=[-0.97, -0.45], norm = [np.cos(np.pi/6), np.sin(np.pi/6)], len = 1))
# scene.objects.append(Wall(center=[0.97, -0.45], norm = [-np.cos(np.pi/6), np.sin(np.pi/6)], len = 1))
scene.objects.append(Wall(center=[0.0, -1.], norm = [0.0, 1.], len = 2))
scene.objects.append(Wall(center=[0.0, 1.], norm = [0.0, -1.], len = 2))
scene.objects.append(Wall(center=[-.7, 0.], norm = [np.cos(np.pi/9), np.sin(np.pi/9)], len = 8))
scene.objects.append(Wall(center=[.7, 0.], norm = [-np.cos(np.pi/9), np.sin(np.pi/9)], len = 8))
# static ball
scene.objects.append(Ball(radius=0.2, static = True, center=[0.3, -0.3], mass = 1e18))
# box
x = 0.3
scene.objects.append(Wall(center=[-0.4, 0.4 + x/2], norm = [0.0, 1.], len = x))
scene.objects.append(Wall(center=[-0.4 + x/2, 0.4], norm = [1.0, 0.], len = x))
scene.objects.append(Wall(center=[-0.4, 0.4 -x/2], norm = [0.0, -1.], len = x))
scene.objects.append(Wall(center=[-0.4 -x/2, 0.4], norm = [-1.0, 0.], len = x))
for obj in scene.objects:
    if obj.static is True:
        obj.vel = [0., 0.]
        obj.accel = np.array([0., 0.])
    else:
        obj.accel = np.array([0., -.5])
scene.show()
app.exec()
