import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

#architecture blocks

blocks = [
    "IoT Devices\n(Smart Meters / Sensors)",
    "Local IDS Training\n(Pytorch Model)",
    "Federated Clients",
    "Federated Server\n(Flower)",
    "Robust Aggregation\n(Median / Trimmed Mean / Krum)",
    "Global IDS Model"
]

y_positions = [5,4,3,2,1,0]

for i, block in enumerate(blocks):
    plt.text(0.5, y_positions[i], block, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="lightblue"))

#arrows
for i in range(len(y_positions)-1):
    plt.arrow(0.5, y_positions[i]-0.2, 0, -0.6, head_width=0.02, head_length=0.1, fc='black')

plt.axis('off')

plt.title("Federated Learning-Based Cyber Attack Detection Architecture")

plt.savefig("results/system_architecture.png")

plt.show()

"""
1. IoT devices such as smart meters generate network traffic.
2. Each devices trains a local IDS model using its own data.
3. Local models send updates to the federated server.
4. The server aggregates model updates using robust aggregation methods.
5. Adversarial attacks are mitigated during aggregation.
6. The final global model is distributed back to clients.
"""