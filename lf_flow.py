import cv2
import numpy as np
import argparse


class OpticalFlowTracker:
    def __init__(self, video_path):
        """
        Initialize tracker object.

        This constructor:
        - Opens the input video source
        - Sets Lucas-Kanade tracking parameters
        - Creates storage for tracked points and previous frame data
        - Generates random colors for visualizing motion trajectories
        """

        # Open video file or webcam source
        self.cap = cv2.VideoCapture(video_path)

        # ---------------- Lucas-Kanade Parameters ----------------
        # Size of square tracking window around every feature point
        self.win_size = 25

        # Maximum iterations allowed while refining motion estimate
        self.max_iters = 20

        # Minimum displacement threshold for convergence
        self.epsilon = 0.01

        # ---------------- Tracking State Variables ----------------
        # Stores currently tracked feature points
        self.points = np.empty((0, 2), dtype=np.float32)

        # Stores previous grayscale frame
        self.old_gray = None

        # Persistent drawing canvas for motion trails
        self.mask = None

        # Random colors assigned to different feature tracks
        self.colors = np.random.randint(0, 255, (300, 3))

    def detect_corners(self, gray, prev_gray=None):
        """
        Detect strong Shi-Tomasi corners for sparse optical flow tracking.

        Logic:
        - Ignore outer image borders (usually noisy / unstable)
        - If previous frame exists, detect moving regions using frame difference
        - Prefer corners inside moving areas
        - Fallback to full image if motion is too small
        """

        h, w = gray.shape
        border_margin = 30

        # Base mask removes border region to avoid unstable edge features
        base_mask = np.zeros((h, w), dtype=np.uint8)
        base_mask[border_margin:h-border_margin, border_margin:w-border_margin] = 255

        if prev_gray is not None:
            # Compute absolute difference between consecutive frames
            diff = cv2.absdiff(gray, prev_gray)

            # Threshold motion regions
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

            # Expand motion regions for stronger feature coverage
            kernel = np.ones((7, 7), np.uint8)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

            # Keep only moving areas inside valid border region
            final_mask = cv2.bitwise_and(base_mask, motion_mask)

            # If too few motion pixels exist, fallback to base mask
            if cv2.countNonZero(final_mask) < 300:
                final_mask = base_mask
        else:
            final_mask = base_mask

        # Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=150,
            qualityLevel=0.01,
            minDistance=7,
            mask=final_mask
        )

        if corners is not None:
            return corners.reshape(-1, 2)

        return np.empty((0, 2), dtype=np.float32)

    def compute_lucas_kanade(self, I1, I2, points):
        """
        Manual implementation of sparse Lucas-Kanade optical flow.

        For each tracked point:
        - Compute spatial gradients Ix and Iy
        - Build LK matrix M
        - Solve displacement iteratively
        - Estimate new point position
        """

        # Gaussian smoothing improves stability for noisy or fast motion
        I1_smooth = cv2.GaussianBlur(I1, (5, 5), 1.5)
        I2_smooth = cv2.GaussianBlur(I2, (5, 5), 1.5)

        half_win = self.win_size // 2
        h, w = I1_smooth.shape

        # Spatial image gradients
        Ix = cv2.Sobel(I1_smooth, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1_smooth, cv2.CV_32F, 0, 1, ksize=3)

        # Output containers
        new_points = np.zeros_like(points, dtype=np.float32)
        status = np.zeros(len(points), dtype=np.uint8)

        for i, pt in enumerate(points):
            x, y = pt[0], pt[1]

            # Skip points too close to image border
            if x < half_win or y < half_win or x >= w - half_win or y >= h - half_win:
                continue

            x_int, y_int = int(round(x)), int(round(y))

            # Extract local gradient window
            ix_win = Ix[y_int-half_win:y_int+half_win+1,
                        x_int-half_win:x_int+half_win+1].flatten()

            iy_win = Iy[y_int-half_win:y_int+half_win+1,
                        x_int-half_win:x_int+half_win+1].flatten()

            # Window must be complete
            if len(ix_win) != self.win_size * self.win_size:
                continue

            # Construct Lucas-Kanade matrix M
            sum_ix2 = np.sum(ix_win * ix_win)
            sum_iy2 = np.sum(iy_win * iy_win)
            sum_ixiy = np.sum(ix_win * iy_win)

            M = np.array([
                [sum_ix2, sum_ixiy],
                [sum_ixiy, sum_iy2]
            ])

            # Reject ill-conditioned windows
            if abs(np.linalg.det(M)) < 1e-5:
                continue

            M_inv = np.linalg.inv(M)

            vx, vy = 0.0, 0.0

            # Reference patch from previous frame
            I1_win = I1_smooth[
                y_int-half_win:y_int+half_win+1,
                x_int-half_win:x_int+half_win+1
            ].flatten()

            # Iterative displacement refinement
            for _ in range(self.max_iters):

                patch2 = cv2.getRectSubPix(
                    I2_smooth,
                    (self.win_size, self.win_size),
                    (x_int + vx, y_int + vy)
                )

                if patch2 is None:
                    break

                I2_win = patch2.flatten()

                # Temporal gradient
                It = I2_win - I1_win

                # Solve velocity update
                b = np.array([
                    -np.sum(ix_win * It),
                    -np.sum(iy_win * It)
                ])

                delta_v = M_inv @ b

                vx += delta_v[0]
                vy += delta_v[1]

                # Stop if displacement becomes very small
                if np.linalg.norm(delta_v) < self.epsilon:
                    break

            # Save valid tracked point
            if not np.isnan(vx) and not np.isnan(vy):
                new_x = x_int + vx
                new_y = y_int + vy

                if 0 <= new_x < w and 0 <= new_y < h:
                    new_points[i] = [new_x, new_y]
                    status[i] = 1

        return new_points, status

    def run(self):
        """
        Main tracking loop.

        Pipeline:
        1. Read frame
        2. Convert to grayscale
        3. Detect scene change
        4. Reinitialize features if needed
        5. Run LK tracking
        6. Draw trajectories
        """

        if not self.cap.isOpened():
            print("Failed to open video")
            return

        ret, old_frame = self.cap.read()
        if not ret:
            return

        # Initialize first grayscale frame
        self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Initial corner detection
        self.points = self.detect_corners(self.old_gray)

        # Empty drawing mask
        self.mask = np.zeros_like(old_frame)

        print("Running Manual LK Tracker (OOP Architecture)... Press 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scene-change detection using frame difference
            diff = cv2.absdiff(gray, self.old_gray)
            mean_diff = np.mean(diff)

            # Reset tracking if major scene change occurs
            if mean_diff > 15.0 or len(self.points) < 10:
                print(f"Background change or lost points. Resetting... (Diff: {mean_diff:.2f})")

                prev = None if mean_diff > 15.0 else self.old_gray

                self.points = self.detect_corners(gray, prev_gray=prev)

                # Clear old trajectories
                self.mask = np.zeros_like(frame)

            elif len(self.points) > 0:

                I1_f32 = self.old_gray.astype(np.float32)
                I2_f32 = gray.astype(np.float32)

                # Compute LK motion
                new_points, status = self.compute_lucas_kanade(I1_f32, I2_f32, self.points)

                # Keep only successful tracks
                good_new = new_points[status == 1]
                good_old = self.points[status == 1]

                # Draw motion vectors
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)

                    color = self.colors[i % len(self.colors)].tolist()

                    self.mask = cv2.line(self.mask, (a, b), (c, d), color, 2)
                    frame = cv2.circle(frame, (a, b), 5, color, -1)

                # Update tracked points
                self.points = good_new

            # Overlay trajectories
            img = cv2.add(frame, self.mask)

            # Display tracked point count
            cv2.putText(
                img,
                f"Points: {len(self.points)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow('Manual LK Tracker (OOP)', img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Update previous frame
            self.old_gray = gray.copy()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LK Sparse Optical Flow")

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file. If omitted, webcam source 0 is used."
    )

    args = parser.parse_args()

    source = args.video if args.video else 0

    tracker = OpticalFlowTracker(source)
    tracker.run()