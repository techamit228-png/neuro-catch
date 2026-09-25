[neuro_catch.py](https://github.com/user-attachments/files/32665606/neuro_catch.py)
# neuro-catch
ии учится ловить мяч
import tkinter as tk
import random
import json
import os
import time

# ============================================================
# NEURO CATCH
# ИИ учится ловить падающие объекты.
#
# Только стандартный Python + tkinter.
#
# Идея:
#   Состояние = положение игрока + положение падающего объекта
#   Действия = LEFT / STAY / RIGHT
#   Награда = + за пойманный объект, - за промах
#
# Чем больше тренировок, тем лучше политика Q-learning.
# Есть SUPER SPEED и NORMAL режимы.
# ============================================================

WIDTH = 760
HEIGHT = 520

PLAYER_W = 90
PLAYER_H = 18
PLAYER_Y = HEIGHT - 55

OBJECT_SIZE = 18
PLAYER_SPEED = 9
OBJECT_SPEED = 5

BG = "#0f1319"
TEXT = "#e9eef7"
MUTED = "#8c97a8"
PLAYER_COLOR = "#4da3ff"
OBJECT_COLOR = "#55d68a"
DANGER_COLOR = "#ff647c"
PANEL = "#181d27"


class Brain:
    def __init__(self):
        self.q = {}

        # Параметры Q-learning
        self.alpha = 0.45
        self.gamma = 0.92
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.99985

        self.episodes = 0
        self.catches = 0
        self.misses = 0
        self.actions_count = 0

    def state(self, player_x, object_x, object_y, player_velocity):
        # Разбиваем пространство на зоны.
        px = max(0, min(15, int(player_x / WIDTH * 16)))
        ox = max(0, min(15, int(object_x / WIDTH * 16)))
        oy = max(0, min(7, int(object_y / HEIGHT * 8)))

        if player_velocity < -2:
            vel = 0
        elif player_velocity > 2:
            vel = 2
        else:
            vel = 1

        # Дополнительный сигнал: объект слева/справа относительно центра игрока.
        delta = object_x - (player_x + PLAYER_W / 2)
        if delta < -90:
            rel = 0
        elif delta < -25:
            rel = 1
        elif delta <= 25:
            rel = 2
        elif delta <= 90:
            rel = 3
        else:
            rel = 4

        return (px, ox, oy, vel, rel)

    def values(self, state):
        if state not in self.q:
            self.q[state] = [0.0, 0.0, 0.0]  # left, stay, right
        return self.q[state]

    def choose(self, state, training=True):
        values = self.values(state)

        if training and random.random() < self.epsilon:
            return random.randrange(3)

        best = max(values)
        best_actions = [i for i, v in enumerate(values) if abs(v - best) < 1e-10]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        values = self.values(state)

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.values(next_state))

        values[action] += self.alpha * (target - values[action])

    def end_episode(self, caught):
        self.episodes += 1
        if caught:
            self.catches += 1
        else:
            self.misses += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, filename="catch_brain.json"):
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes": self.episodes,
            "catches": self.catches,
            "misses": self.misses,
            "q": [[list(k), v] for k, v in self.q.items()]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filename="catch_brain.json"):
        if not os.path.exists(filename):
            return False

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.episodes = data.get("episodes", 0)
        self.catches = data.get("catches", 0)
        self.misses = data.get("misses", 0)

        self.q = {}
        for item in data.get("q", []):
            key, values = item
            self.q[tuple(key)] = values

        return True


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURO CATCH — ИИ учится ловить")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.brain = Brain()

        self.player_x = WIDTH / 2 - PLAYER_W / 2
        self.object_x = 0
        self.object_y = 0

        self.last_player_x = self.player_x
        self.running_training = False
        self.running_demo = False

        self.fast = True
        self.demo_delay = 0.03

        self.recent = []
        self.episode_reward = 0.0

        self.build()
        self.spawn()

        try:
            self.brain.load()
        except Exception:
            pass

        self.draw()
        self.stats()

    def build(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(padx=14, pady=14)

        tk.Label(
            outer,
            text="🧠 NEURO CATCH",
            font=("Segoe UI", 20, "bold"),
            bg=BG,
            fg=TEXT
        ).pack(anchor="w")

        tk.Label(
            outer,
            text="ИИ учится ловить падающие объекты методом Q-learning",
            font=("Segoe UI", 9),
            bg=BG,
            fg=MUTED
        ).pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(
            outer,
            width=WIDTH,
            height=HEIGHT,
            bg=BG,
            highlightthickness=0
        )
        self.canvas.pack()

        panel = tk.Frame(outer, bg=PANEL)
        panel.pack(fill="x", pady=(10, 0))

        row = tk.Frame(panel, bg=PANEL)
        row.pack(padx=8, pady=8, fill="x")

        controls = [
            ("⚡ СКОРОСТЬ", self.toggle_speed),
            ("▶ ОБУЧАТЬ", self.toggle_training),
            ("👁 ДЕМО", self.demo),
            ("🎯 НОВАЯ СЕРИЯ", self.new_series),
            ("💾 СОХРАНИТЬ", self.save),
            ("📂 ЗАГРУЗИТЬ", self.load),
            ("🧹 СБРОСИТЬ", self.reset_brain),
        ]

        for name, cmd in controls:
            tk.Button(
                row,
                text=name,
                command=cmd,
                bg="#252c39",
                fg=TEXT,
                activebackground="#343d4d",
                activeforeground="white",
                relief="flat",
                bd=0,
                padx=8,
                pady=6,
                font=("Segoe UI", 8, "bold")
            ).pack(side="left", padx=2)

        self.mode_label = tk.Label(
            panel,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=("Consolas", 9)
        )
        self.mode_label.pack(anchor="w", padx=12)

        self.stat_label = tk.Label(
            panel,
            text="",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            font=("Consolas", 10)
        )
        self.stat_label.pack(anchor="w", padx=12, pady=(2, 10))

        self.status_label = tk.Label(
            outer,
            text="Готов.",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 9)
        )
        self.status_label.pack(anchor="w", pady=(7, 0))

    def spawn(self):
        self.object_x = random.randint(12, WIDTH - OBJECT_SIZE - 12)
        self.object_y = -OBJECT_SIZE
        self.episode_reward = 0.0

    def reset_round(self):
        self.player_x = WIDTH / 2 - PLAYER_W / 2
        self.last_player_x = self.player_x
        self.spawn()

    def draw(self):
        self.canvas.delete("all")

        # Линии-индикаторы
        for i in range(1, 8):
            y = i * HEIGHT / 8
            self.canvas.create_line(0, y, WIDTH, y, fill="#171d26")

        # Зона цели
        self.canvas.create_rectangle(
            0, PLAYER_Y + PLAYER_H + 8,
            WIDTH, HEIGHT,
            fill="#141a22",
            outline=""
        )

        # Падающий объект
        self.canvas.create_oval(
            self.object_x,
            self.object_y,
            self.object_x + OBJECT_SIZE,
            self.object_y + OBJECT_SIZE,
            fill=OBJECT_COLOR,
            outline=""
        )

        # Игрок
        self.canvas.create_rectangle(
            self.player_x,
            PLAYER_Y,
            self.player_x + PLAYER_W,
            PLAYER_Y + PLAYER_H,
            fill=PLAYER_COLOR,
            outline=""
        )

        # Центр игрока
        center = self.player_x + PLAYER_W / 2
        self.canvas.create_line(
            center, PLAYER_Y - 10,
            center, PLAYER_Y + PLAYER_H + 10,
            fill="#a9cfff"
        )

        # Подписи
        self.canvas.create_text(
            12, 14,
            anchor="nw",
            text="Падающий объект",
            fill=MUTED,
            font=("Segoe UI", 9)
        )
        self.canvas.create_text(
            12, HEIGHT - 25,
            anchor="sw",
            text="Платформа ИИ",
            fill=MUTED,
            font=("Segoe UI", 9)
        )

        # Рекомендованное действие
        state = self.brain.state(
            self.player_x,
            self.object_x,
            self.object_y,
            self.player_x - self.last_player_x
        )
        values = self.brain.values(state)
        best = max(values)
        best_actions = [
            "← СЛЕВА",
            "• ОСТАТЬСЯ",
            "→ СПРАВА"
        ]
        idx = values.index(best)

        self.canvas.create_text(
            WIDTH - 10,
            14,
            anchor="ne",
            text=f"Решение: {best_actions[idx]}",
            fill=TEXT,
            font=("Consolas", 10, "bold")
        )

    def step_environment(self, training):
        old_state = self.brain.state(
            self.player_x,
            self.object_x,
            self.object_y,
            self.player_x - self.last_player_x
        )

        action = self.brain.choose(old_state, training=training)

        self.last_player_x = self.player_x

        if action == 0:
            self.player_x -= PLAYER_SPEED
        elif action == 2:
            self.player_x += PLAYER_SPEED

        self.player_x = max(0, min(WIDTH - PLAYER_W, self.player_x))

        # Чем выше объект, тем больше движения по вертикали.
        self.object_y += OBJECT_SPEED

        caught = False
        done = False

        # Проверка столкновения.
        if (
            self.object_y + OBJECT_SIZE >= PLAYER_Y
            and self.object_y <= PLAYER_Y + PLAYER_H
            and self.object_x + OBJECT_SIZE >= self.player_x
            and self.object_x <= self.player_x + PLAYER_W
        ):
            caught = True
            done = True

        elif self.object_y > HEIGHT:
            done = True

        # Небольшая shaped-награда:
        # ИИ получает подсказку, когда приближается к правильной позиции.
        object_center = self.object_x + OBJECT_SIZE / 2
        player_center = self.player_x + PLAYER_W / 2
        distance = abs(object_center - player_center)

        proximity = max(0.0, 1.0 - distance / WIDTH)
        reward = proximity * 0.15

        if caught:
            reward += 35.0
        elif done:
            reward -= 25.0

        next_state = self.brain.state(
            self.player_x,
            self.object_x,
            self.object_y,
            self.player_x - self.last_player_x
        )

        if training:
            self.brain.learn(old_state, action, reward, next_state, done)
            self.brain.actions_count += 1

        self.episode_reward += reward

        if done:
            if training:
                self.brain.end_episode(caught)

            self.recent.append(1 if caught else 0)
            self.recent = self.recent[-100:]
            self.spawn()

        return caught, done

    def training_loop(self):
        if not self.running_training:
            return

        # При ускорении делаем большой пакет шагов.
        batch = 4000 if self.fast else 1

        for _ in range(batch):
            self.step_environment(training=True)

        self.draw()
        self.stats()

        self.root.after(1 if self.fast else 25, self.training_loop)

    def demo(self):
        self.running_training = False
        self.running_demo = True
        self.reset_round()
        self.status_label.config(
            text="👁 ДЕМО: ИИ не обучается, а использует накопленные знания.",
            fg=PLAYER_COLOR
        )
        self.demo_loop()

    def demo_loop(self):
        if not self.running_demo:
            return

        self.step_environment(training=False)

        self.draw()
        self.stats()

        self.root.after(
            max(1, int(self.demo_delay * 1000)),
            self.demo_loop
        )

    def toggle_training(self):
        self.running_demo = False
        self.running_training = not self.running_training

        if self.running_training:
            self.status_label.config(
                text="🧠 ИИ ОБУЧАЕТСЯ... Нажми кнопку ещё раз для остановки.",
                fg=OBJECT_COLOR
            )
            self.training_loop()
        else:
            self.status_label.config(
                text="Обучение остановлено. Можно включить ДЕМО.",
                fg=TEXT
            )

    def toggle_speed(self):
        self.fast = not self.fast

        if self.fast:
            self.demo_delay = 0.03
            self.status_label.config(
                text="⚡ СУПЕР-СКОРОСТЬ: огромное количество тренировочных шагов.",
                fg=OBJECT_COLOR
            )
        else:
            self.demo_delay = 0.12
            self.status_label.config(
                text="🐢 НОРМАЛЬНАЯ СКОРОСТЬ: можно видеть обучение почти в реальном времени.",
                fg=TEXT
            )

        self.update_mode()

    def update_mode(self):
        if self.fast:
            self.mode_label.config(
                text="Режим: ⚡ СУПЕР-СКОРОСТЬ | обучение пакетами"
            )
        else:
            self.mode_label.config(
                text="Режим: 🐢 НОРМАЛЬНЫЙ | медленные отдельные шаги"
            )

    def new_series(self):
        self.running_training = False
        self.running_demo = False
        self.reset_round()

        self.status_label.config(
            text="🎯 Новая серия объектов.",
            fg=TEXT
        )
        self.draw()

    def save(self):
        try:
            self.brain.save()
            self.status_label.config(
                text=f"💾 Мозг сохранён в catch_brain.json ({len(self.brain.q):,} состояний).",
                fg=OBJECT_COLOR
            )
        except Exception as e:
            self.status_label.config(
                text=f"Ошибка сохранения: {e}",
                fg=DANGER_COLOR
            )

    def load(self):
        try:
            if self.brain.load():
                self.status_label.config(
                    text=f"📂 Мозг загружен. Эпизодов: {self.brain.episodes:,}.",
                    fg=OBJECT_COLOR
                )
                self.draw()
                self.stats()
            else:
                self.status_label.config(
                    text="Файл catch_brain.json ещё не создан.",
                    fg=TEXT
                )
        except Exception as e:
            self.status_label.config(
                text=f"Ошибка загрузки: {e}",
                fg=DANGER_COLOR
            )

    def reset_brain(self):
        self.running_training = False
        self.running_demo = False
        self.brain = Brain()
        self.recent.clear()
        self.reset_round()

        self.status_label.config(
            text="🧹 Память ИИ очищена. Он снова ничего не умеет.",
            fg=TEXT
        )

        self.draw()
        self.stats()

    def stats(self):
        episodes = self.brain.episodes
        catches = self.brain.catches

        overall = catches / episodes * 100 if episodes else 0.0
        recent = sum(self.recent) / len(self.recent) * 100 if self.recent else 0.0

        self.stat_label.config(
            text=(
                f"Эпизоды: {episodes:,}    "
                f"Поймано: {catches:,}    "
                f"Промахов: {self.brain.misses:,}\n"
                f"Общий успех: {overall:6.2f}%    "
                f"Последние 100: {recent:6.2f}%    "
                f"ε: {self.brain.epsilon:.4f}    "
                f"Состояний: {len(self.brain.q):,}"
            )
        )
        self.update_mode()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
