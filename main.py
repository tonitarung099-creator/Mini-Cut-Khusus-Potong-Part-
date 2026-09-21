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
    elif "--self-test-layout" in sys.argv:
        def verify_layout():
            win.resize(1120, 680)
            app.processEvents()
            issues = win.ui_layout_issues()
            if issues:
                for issue in issues:
                    print("layout-failed: " + issue)
                app.exit(4)
            else:
                print("layout-ok: no overlapping or clipped interactive controls")
                app.exit(0)
        QTimer.singleShot(1600, verify_layout)
    elif "--self-test" in sys.argv:
        QTimer.singleShot(1200, app.quit)
    raise SystemExit(app.exec())

if __name__ == "__main__":
    main()
