import pydobot
import time

# Connect to the Dobot
device = pydobot.Dobot(port="COM8")
device.speed(150, 150)

# Open gripper at the start (make sure nothing is held)
device.grip(False)
time.sleep(0.5)

# Safety height to move up/down (z)
safe_z = 0   # don't change
place_z = -10

# Positions format: [x, y, z, r]
# First four = Pallet A (1..4)

palA_positions = [
    [270.73, -17.56, -44.75, 0],   # 1
    [329.19, -17.44, -43.77, 0],   # 2
    [270.33, -75.43, -44.66, 0],  # 3
    [329.19, -77.88, -46.51, 0],  # 4
]

# Next four = Pallet B (5..8) arranged so A1->B1(=5th overall), A2->B2(=6th), A3->B3(=7th), A4->B4(=8th)
palB_positions = [
    [270.74,  74.4, -44.80, 0],   # 5 - FIXED: was 268.74, now 273.74
    [330.23,  73.37, -46.57, 0],  # 6 - FIXED: was 328.23, now 333.23
    [270.93,  15.95, -44.77, 0],  # 7 - FIXED: was 268.93, now 273.93
    [329.46,  13.8, -44.87, 0],   # 8 - FIXED: was 326.86, now 331.86
]

# ---------------- Helper function ----------------


def pick_and_place(pick, place):
    px, py, pz, pr = pick
    dx, dy, dz, dr = place

    # ---- Pick ----
    device.move_to(x=px, y=py, z=safe_z, r=pr)
    time.sleep(2)
    device.move_to(x=px, y=py, z=place_z, r=pr)
    device.grip(True)      # Close gripper
    time.sleep(3)
    device.move_to(x=px, y=py, z=safe_z, r=pr)
    time.sleep(1)

    # ---- Place ----
    device.move_to(x=dx, y=dy, z=safe_z, r=dr)
    time.sleep(0.5)
    device.move_to(x=dx, y=dy, z=place_z, r=dr)
    device.grip(False)     # Open gripper
    time.sleep(0.8)
    device.move_to(x=dx, y=dy, z=safe_z, r=dr)
    time.sleep(0.5)


# ---------------- Main Loop ----------------
# Forward trip: A -> B
for i in range(len(palA_positions)):
    pick_and_place(palA_positions[i], palB_positions[i])

# Return trip: B -> A (reverse order so no block is skipped)
for i in reversed(range(len(palB_positions))):
    pick_and_place(palB_positions[i], palA_positions[i])

# 🔹 Ensure gripper is open after the last block
device.grip(False)
time.sleep(0.5)

print("Task completed: all blocks moved A→B and back B→A, gripper open.")
