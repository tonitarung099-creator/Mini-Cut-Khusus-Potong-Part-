import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from minicut_agent.ui import MiniCutWindow

def main():
    app = QApplication(sys.argv)
    win = MiniCutWindow()
    win.show()
    if "--self-test-player" in sys.argv:
        def verify_player_backend():
            if win.player.using_mpv:
                print("player-backend-ok: mpv")
                app.exit(0)
            else:
                print("player-backend-failed: " + win.player.backend_name)
                app.exit(3)
        QTimer.singleShot(1800, verify_player_backend)
    elif "--self-test" in sys.argv:
        QTimer.singleShot(1200, app.quit)
    raise SystemExit(app.exec())

if __name__ == "__main__":
    main()
