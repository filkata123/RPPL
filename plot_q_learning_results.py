import matplotlib.pyplot as plt

data = []
with open("qlearning_path_log.txt") as f:
    for line in f:
        ep, plen = line.strip().split(",")
        if plen != "None":
            data.append((int(ep), int(plen)))

episodes, lengths = zip(*data)
plt.plot(episodes, lengths)
plt.xlabel("Episode")
plt.ylabel("Path Length")
plt.title("Q-learning Path Length Over Time")
plt.grid(True)
plt.show()
