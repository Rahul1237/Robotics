"""
TIC-TAC-TOE ROBOT with DOBOT ARM + MINIMAX + OPENCV
Human places blocks manually, Robot uses Dobot arm to place blocks
WINDOWS VERSION - Uses COM11 port (detected from your system)
Robot returns to home position after each move for vision capture
"""
import cv2
import numpy as np
import time
import pydobot


class RoboticTicTacToe:
    def __init__(self, dobot_port="COM11", camera_index=1):
        """Initialize the Robotic Tic-Tac-Toe game"""
        print("[INIT] Connecting to Dobot...")
        self.device = pydobot.Dobot(port=dobot_port)
        self.device.speed(150, 150)
        self.device.suck(False)
        time.sleep(0.5)
        print("[OK] Dobot connected!")
        
        self.camera_index = camera_index
        self.board = np.zeros((3, 3), dtype=int)
        self.EMPTY = 0
        self.HUMAN = 1
        self.ROBOT = 2
        
        # Z-axis heights
        self.safe_z = 0
        self.place_z = -35.5
        
        # Home position for vision capture
        self.home_position = [250.0, 0.0, 50.0, 0]
        
        # 4 PICKUP SPOTS - where RED blocks are stored
        self.pickup_spots = [
            [270.9, -150.4, -49.8, 0],
            [291.9, -133.8, -49.0, 0],
            [310.8, -152.5, -48.8, 0],
            [327.9, -131.3, -48.9, 0],
        ]
        self.current_pickup_index = 0
        
        # 3x3 GRID POSITIONS - where blocks are placed
        self.grid_positions = {
            (0, 0): [324.4, -57.1, -35.9, 0],
            (0, 1): [324.4, -28.0, -35.9, 0],
            (0, 2): [324.4, -0.9,  -35.9, 0],
            (1, 0): [300.0, -53.8, -35.9, 0],
            (1, 1): [300.7, -24.4, -35.9, 0],
            (1, 2): [300.0, -3.8,  -35.9, 0],
            (2, 0): [277.4, -56.3, -35.9, 0],
            (2, 1): [277.9, -29.8, -35.9, 0],
            (2, 2): [279.2, -3.8,  -35.9, 0],
        }
        
        # Color detection HSV ranges
        self.human_lower = np.array([100, 100, 100])
        self.human_upper = np.array([130, 255, 255])
        self.robot_lower1 = np.array([0, 100, 100])
        self.robot_upper1 = np.array([10, 255, 255])
        self.robot_lower2 = np.array([160, 100, 100])
        self.robot_upper2 = np.array([180, 255, 255])
        
        # Winning combinations
        self.winning_combos = [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        ]
        
        # Game state
        self.scores = {'human': 0, 'robot': 0, 'draws': 0}
        self.last_move_time = 0
        self.move_cooldown = 2.0
        
        # Move to home position on startup
        self.move_to_home()
    
    def move_to_home(self):
        """Move robot to home position for vision capture"""
        print("[ROBOT] Moving to HOME position for vision capture...")
        self.device.move_to(x=self.home_position[0], y=self.home_position[1], 
                           z=self.home_position[2], r=self.home_position[3])
        time.sleep(1.0)
        print("[ROBOT] At HOME position - ready for vision capture")
    
    def pick_and_place(self, pickup, place):
        """Pick block from pickup spot and place it at target position"""
        px, py, pz, pr = pickup
        dx, dy, dz, dr = place
        
        print(f"[ROBOT] Moving to pickup ({px:.1f}, {py:.1f})")
        self.device.move_to(x=px, y=py, z=self.safe_z, r=pr)
        time.sleep(0.5)
        
        self.device.move_to(x=px, y=py, z=pz, r=pr)
        self.device.suck(True)
        time.sleep(1.5)
        print("[ROBOT] Block grabbed!")
        
        self.device.move_to(x=px, y=py, z=self.safe_z, r=pr)
        time.sleep(0.5)
        
        print(f"[ROBOT] Moving to grid position ({dx:.1f}, {dy:.1f})")
        self.device.move_to(x=dx, y=dy, z=self.safe_z, r=dr)
        time.sleep(0.5)
        
        self.device.move_to(x=dx, y=dy, z=dz, r=dr)
        self.device.suck(False)
        time.sleep(1.5)
        print("[ROBOT] Block placed!")
        
        self.device.move_to(x=dx, y=dy, z=self.safe_z, r=dr)
        time.sleep(0.5)
    
    def robot_place_block(self, row, col):
        """Robot picks RED block from pickup spot and places at grid position"""
        print(f"\n{'='*60}")
        print(f"[ROBOT TURN] Placing RED block at grid ({row},{col})")
        print(f"{'='*60}")
        
        pickup_pos = self.pickup_spots[self.current_pickup_index]
        print(f"[ROBOT] Using pickup spot {self.current_pickup_index + 1}: ({pickup_pos[0]:.1f}, {pickup_pos[1]:.1f})")
        self.current_pickup_index = (self.current_pickup_index + 1) % len(self.pickup_spots)
        
        target_pos = self.grid_positions[(row, col)]
        self.pick_and_place(pickup_pos, target_pos)
        
        print("[ROBOT] RED block placed!")
        print(f"{'='*60}\n")
        
        # Return to home position for vision
        self.move_to_home()
    
    def detect_grid(self, frame):
        """Detect the 3x3 grid in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 10000:
                return approx.reshape(4, 2)
        return None
    
    def order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def get_perspective_transform(self, frame, grid_corners):
        """Transform grid to top-down view"""
        if grid_corners is None:
            return None
        pts = self.order_points(grid_corners)
        dst = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(frame, M, (450, 450))
    
    def detect_block_in_cell(self, cell_image):
        """Detect if cell contains BLUE (human) or RED (robot) block"""
        hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.human_lower, self.human_upper)
        red_mask1 = cv2.inRange(hsv, self.robot_lower1, self.robot_upper1)
        red_mask2 = cv2.inRange(hsv, self.robot_lower2, self.robot_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        blue_ratio = cv2.countNonZero(blue_mask) / (cell_image.shape[0] * cell_image.shape[1])
        red_ratio = cv2.countNonZero(red_mask) / (cell_image.shape[0] * cell_image.shape[1])
        
        threshold = 0.10
        if blue_ratio > threshold and blue_ratio > red_ratio:
            return self.HUMAN
        elif red_ratio > threshold:
            return self.ROBOT
        else:
            return self.EMPTY
    
    def detect_board_state(self, warped_grid):
        """Detect all BLUE and RED blocks on the board"""
        cell_w = warped_grid.shape[1] // 3
        cell_h = warped_grid.shape[0] // 3
        detected_board = np.zeros((3, 3), dtype=int)
        
        for i in range(3):
            for j in range(3):
                cell = warped_grid[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                detected_color = self.detect_block_in_cell(cell)
                detected_board[i][j] = detected_color if self.board[i][j] == self.EMPTY else self.board[i][j]
        
        return detected_board
    
    def check_winner(self, board):
        """Check if there's a winner"""
        for combo in self.winning_combos:
            values = [board[a][b] for a, b in combo]
            if values[0] != self.EMPTY and values[0] == values[1] == values[2]:
                return values[0]
        return None
    
    def is_board_full(self, board):
        """Check if board is full"""
        return not np.any(board == self.EMPTY)
    
    def get_available_moves(self, board):
        """Get list of available positions"""
        return [(i, j) for i in range(3) for j in range(3) if board[i][j] == self.EMPTY]
    
    def minimax(self, board, depth, is_max, alpha=-float('inf'), beta=float('inf')):
        """Minimax algorithm with Alpha-Beta pruning"""
        winner = self.check_winner(board)
        if winner == self.ROBOT:
            return 10 - depth
        elif winner == self.HUMAN:
            return depth - 10
        elif self.is_board_full(board):
            return 0
        
        if is_max:
            max_eval = -float('inf')
            for move in self.get_available_moves(board):
                board[move[0]][move[1]] = self.ROBOT
                val = self.minimax(board, depth+1, False, alpha, beta)
                board[move[0]][move[1]] = self.EMPTY
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_available_moves(board):
                board[move[0]][move[1]] = self.HUMAN
                val = self.minimax(board, depth+1, True, alpha, beta)
                board[move[0]][move[1]] = self.EMPTY
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_best_move(self):
        """Get the best move for AI using Minimax"""
        print("\n[AI] Calculating best move with Minimax...")
        start_time = time.time()
        
        best_score = -float('inf')
        best_move = None
        
        for move in self.get_available_moves(self.board):
            self.board[move[0]][move[1]] = self.ROBOT
            score = self.minimax(self.board, 0, False)
            self.board[move[0]][move[1]] = self.EMPTY
            
            print(f"  Cell ({move[0]},{move[1]}): Score = {score}")
            
            if score > best_score:
                best_score = score
                best_move = move
        
        elapsed = time.time() - start_time
        print(f"[AI] Best move: ({best_move[0]},{best_move[1]}) | Score: {best_score} | Time: {elapsed:.3f}s")
        
        return best_move
    
    def visualize_board(self, frame, warped_grid):
        """Draw detected board state"""
        if warped_grid is None:
            return frame
        
        cw, ch = warped_grid.shape[1] // 3, warped_grid.shape[0] // 3
        overlay = warped_grid.copy()
        
        # Draw grid lines
        for i in range(1, 3):
            cv2.line(overlay, (i*cw, 0), (i*cw, overlay.shape[0]), (0, 255, 0), 3)
            cv2.line(overlay, (0, i*ch), (overlay.shape[1], i*ch), (0, 255, 0), 3)
        
        # Draw symbols
        for i in range(3):
            for j in range(3):
                cx = j * cw + cw // 2
                cy = i * ch + ch // 2
                if self.board[i][j] == self.HUMAN:
                    # Draw X for human (BLUE)
                    cv2.line(overlay, (cx-30, cy-30), (cx+30, cy+30), (255, 0, 0), 5)
                    cv2.line(overlay, (cx+30, cy-30), (cx-30, cy+30), (255, 0, 0), 5)
                elif self.board[i][j] == self.ROBOT:
                    # Draw O for robot (RED)
                    cv2.circle(overlay, (cx, cy), 35, (0, 0, 255), 5)
        
        return overlay
    
    def print_board_console(self):
        """Print board to console"""
        print("\nCurrent Board:")
        symbols = {self.EMPTY: '.', self.HUMAN: 'B', self.ROBOT: 'R'}
        for i in range(3):
            row = ' | '.join([symbols[self.board[i][j]] for j in range(3)])
            print(f"  {row}")
            if i < 2:
                print("  " + "-"*9)
    
    def reset_game(self):
        """Reset the board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.last_move_time = 0
        self.current_pickup_index = 0
        self.device.suck(False)
        time.sleep(0.5)
        self.move_to_home()
    
    def cleanup(self):
        """Cleanup before exit"""
        self.device.suck(False)
        time.sleep(0.5)
        print("[CLEANUP] Dobot released")


def check_com_ports():
    """Check available COM ports"""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if ports:
            print("\n[INFO] Available COM Ports:")
            for port in ports:
                print(f"  - {port.device}: {port.description}")
            return ports
        else:
            print("\n[WARNING] No COM ports detected!")
            return []
    except Exception as e:
        print(f"[WARNING] Could not list COM ports: {e}")
        return []


def main():
    """Main function to run the Tic-Tac-Toe robot game"""
    print("\n" + "="*60)
    print("TIC-TAC-TOE ROBOT - DOBOT + MINIMAX + OPENCV (WINDOWS)")
    print("="*60)
    
    available_ports = check_com_ports()
    
    # Configuration for Windows - UPDATED TO COM11
    camera_index = 1   # External USB camera
    dobot_port = "COM11"  # Detected from your system: USB Serial Device
    
    print(f"\n[CONFIG] Using Camera Index: {camera_index}")
    print(f"[CONFIG] Using Dobot Port: {dobot_port}")
    print(f"[INFO] Detected USB Serial Device on {dobot_port}")
    
    try:
        game = RoboticTicTacToe(dobot_port=dobot_port, camera_index=camera_index)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize robot: {e}")
        print("\nTroubleshooting:")
        print("  1. Check Dobot is connected and powered on")
        print("  2. Try unplugging and replugging USB cable")
        print("  3. Check Device Manager for correct COM port")
        print("  4. Install Dobot drivers: https://www.dobot.cc/downloadcenter.html")
        print("\nAvailable ports detected:")
        for port in available_ports:
            print(f"  - {port.device}: {port.description}")
        return
    
    print("\n" + "="*60)
    print("GAME RULES:")
    print("="*60)
    print("1. Draw a 3x3 grid on paper/mat")
    print("2. YOU (Human): Place BLUE blocks manually on the grid")
    print("3. ROBOT: Picks RED blocks and places using Minimax AI")
    print("4. Camera detects both BLUE and RED blocks automatically")
    print("5. Robot returns to HOME position after each move for vision")
    print("\nControls:")
    print("  'R' - Reset game")
    print("  'Q' - Quit")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}!")
        print("[TIP] Run the camera test script first to find correct camera_index")
        game.cleanup()
        return
    
    print(f"[OK] Camera {camera_index} opened successfully!")
    
    game_active = True
    robot_move_pending = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame")
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            grid_corners = game.detect_grid(frame)
            
            if grid_corners is not None:
                cv2.polylines(display_frame, [grid_corners], True, (0, 255, 0), 3)
                cv2.putText(display_frame, "GRID DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                warped = game.get_perspective_transform(frame, grid_corners)
                
                if warped is not None:
                    detected_board = game.detect_board_state(warped)
                    current_time = time.time()
                    
                    # Check for new human move
                    if game_active and not robot_move_pending and current_time - game.last_move_time > game.move_cooldown:
                        for i in range(3):
                            for j in range(3):
                                if game.board[i][j] == game.EMPTY and detected_board[i][j] == game.HUMAN:
                                    print(f"\n{'='*50}")
                                    print(f"[DETECTED] BLUE block (Human) placed at ({i},{j})")
                                    print(f"{'='*50}")
                                    game.board[i][j] = game.HUMAN
                                    game.print_board_console()
                                    game.last_move_time = current_time
                                    
                                    winner = game.check_winner(game.board)
                                    if winner == game.HUMAN:
                                        print("\n" + "="*50)
                                        print("[WIN] HUMAN WINS!")
                                        print("="*50)
                                        game.scores['human'] += 1
                                        game_active = False
                                    elif game.is_board_full(game.board):
                                        print("\n" + "="*50)
                                        print("[DRAW] It's a tie!")
                                        print("="*50)
                                        game.scores['draws'] += 1
                                        game_active = False
                                    else:
                                        robot_move_pending = True
                                    break
                            if robot_move_pending:
                                break
                    
                    # Robot's turn
                    if robot_move_pending and game_active:
                        best_move = game.get_best_move()
                        if best_move:
                            game.robot_place_block(best_move[0], best_move[1])
                            game.board[best_move[0]][best_move[1]] = game.ROBOT
                            game.print_board_console()
                            
                            winner = game.check_winner(game.board)
                            if winner == game.ROBOT:
                                print("\n" + "="*50)
                                print("[WIN] ROBOT WINS!")
                                print("="*50)
                                game.scores['robot'] += 1
                                game_active = False
                            elif game.is_board_full(game.board):
                                print("\n" + "="*50)
                                print("[DRAW] It's a tie!")
                                print("="*50)
                                game.scores['draws'] += 1
                                game_active = False
                            
                            robot_move_pending = False
                            game.last_move_time = time.time()
                    
                    # Visualize
                    visual = game.visualize_board(display_frame, warped)
                    cv2.imshow("Detected Grid", visual)
            else:
                cv2.putText(display_frame, "NO GRID DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show scores
            score_text = f"Human: {game.scores['human']} | Robot: {game.scores['robot']} | Draws: {game.scores['draws']}"
            cv2.putText(display_frame, score_text, (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Tic-Tac-Toe Robot", display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif key in [ord('r'), ord('R')]:
                print("\n[RESET] New game started!")
                game.reset_game()
                game_active = True
                robot_move_pending = False
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping...")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
    finally:
        game.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[END] Game finished. Final scores:")
        print(f"  Human: {game.scores['human']}")
        print(f"  Robot: {game.scores['robot']}")
        print(f"  Draws: {game.scores['draws']}")
        print("\nThanks for playing!\n")


if __name__ == "__main__":
    main()