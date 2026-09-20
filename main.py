import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from minicut_agent.ui import MiniCutWindow

def main():
    app = QApplication(sys.argv)
    win = MiniCutWindow()
    win.show()
    if "--self-test" in sys.argv:
        QTimer.singleShot(1200, app.quit)
    raise SystemExit(app.exec())

if __name__ == "__main__":
    main()
