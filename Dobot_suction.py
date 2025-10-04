import pydobot
import time

# Connect to the Dobot
device = pydobot.Dobot(port="COM3")
device.speed(150, 150)

# Open gripper at the start (make sure nothing is held)
device.suck(False)
time.sleep(0.4)

# Safety height to move up/down (z)
safe_z = 0   # don't change
place_z = -44

# Positions format: [x, y, z, r]
# First four = Pallet A (1..4)

palA_positions = [
    [270.73, -17.56, -44.45, 0],   # 1
    [329.19, -17.44, -42.67, 0],   # 2
    [270.33, -75.43, -44.26, 0],  # 3
    [329.19, -77.88, -46.11, 0],  # 4
]

# Next four = Pallet B (5..8) arranged so A1->B1(=5th overall), A2->B2(=6th), A3->B3(=7th), A4->B4(=8th)
palB_positions = [
    [268.74,  72.4, -44.40, 0],  # 5  (for A1)
    [328.23,  71.37, -46.17, 0],  # 6  (for A2)
    [268.93,  14.95, -44.37, 0],  # 7  (for A3)
    [326.86,  12.8, -44.97, 0],  # 8  (for A4)
]

# ---------------- Helper function ----------------


def pick_and_place(pick, place):
    px, py, pz, pr = pick
    dx, dy, dz, dr = place

    # ---- Pick ----
    device.move_to(x=px, y=py, z=safe_z, r=pr)
    time.sleep(0.5)
    device.move_to(x=px, y=py, z=place_z, r=pr)
    device.suck(True)      # Close gripper
    time.sleep(1)
    device.move_to(x=px, y=py, z=safe_z, r=pr)
    time.sleep(0.3)

    # ---- Place ----
    device.move_to(x=dx, y=dy, z=safe_z, r=dr)
    time.sleep(0.5)
    device.move_to(x=dx, y=dy, z=place_z, r=dr)
    device.suck(False)     # Open gripper
    time.sleep(1)
    device.move_to(x=dx, y=dy, z=safe_z, r=dr)
    time.sleep(0.3)


# ---------------- Main Loop ----------------
# Forward trip: A -> B
for i in range(len(palA_positions)):
    pick_and_place(palA_positions[i], palB_positions[i])

# Return trip: B -> A (reverse order so no block is skipped)
for i in reversed(range(len(palB_positions))):
    pick_and_place(palB_positions[i], palA_positions[i])

# 🔹 Ensure gripper is open after the last block
device.suck(False)
time.sleep(0.5)

print("✅ Task completed: all blocks moved A→B and back B→A, gripper open.")