# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5
File Name: window.py
Author: luguodexiaopan
Create Date: 2023/4/26
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from subprocess import call

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

torch.cuda.is_available()


# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self, parent=None):
        # 初始化界面
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('课堂违纪检测系统')
        self.resize(1600, 900)
        self.setWindowIcon(QIcon("images/UI/you.jpg"))
        self.initUI()

    '''
    ***背景图片设置***
    '''
    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        pixmap = QPixmap("./images/background/落日.jpg")
        painter.drawPixmap(self.rect(), pixmap)

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 44)
        font_main = QFont('楷体', 20)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("欢  迎  来  到  课  堂  违  纪  检  测  系  统")
        img_detection_title.setAlignment(Qt.AlignCenter)
        img_detection_title.setFont(font_title)
        img_detection_title.setStyleSheet("QLabel{color:black}"
                                          "QLabel{text-shadow: 0 0 15px #0080FF,0 0 15px #7F00FF,0 0 15px #FF0000;}")
        mid_img_widget = QWidget()


        # mid_img_layout = QHBoxLayout()
        # self.left_img = QLabel()
        # self.right_img = QLabel()
        # self.left_img.setPixmap(QPixmap("images/UI/shoe.jpg"))
        # self.right_img.setPixmap(QPixmap("images/UI/cell_phone.jpg"))
        # self.left_img.setAlignment(Qt.AlignCenter)
        # self.right_img.setAlignment(Qt.AlignCenter)
        # mid_img_layout.addWidget(self.left_img)
        # mid_img_layout.addStretch(0)
        # mid_img_layout.addWidget(self.right_img)
        # mid_img_widget.setLayout(mid_img_layout)


        shoe = QPushButton("— — > >  鞋 类 检 测  < < — —")
        phone = QPushButton("— — > >  手 机 检 测  < < — —")
        shoe.clicked.connect(self.run_script1)
        phone.clicked.connect(self.run_script2)
        shoe.setFont(font_main)
        phone.setFont(font_main)
        shoe.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgba(220,220,220,0.1)}"
                                    "QPushButton{background-color:transparent;}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-style:none;}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:20px 10px}"
                                    "QPushButton{margin:5px 5px}")
        phone.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgba(220,220,220,0.1)}"
                                     "QPushButton{background-color:transparent;}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-style:none;}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:20px 10px}"
                                     "QPushButton{margin:5px 5px 50px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(shoe)
        img_detection_layout.addWidget(phone)
        img_detection_widget.setLayout(img_detection_layout)

        self.addTab(img_detection_widget, '系统主页')
        self.setTabIcon(0, QIcon('images/UI/you.jpg'))


    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否关闭当前窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 跳转到鞋类检测.py文件 ### 
    '''

    def run_script1(self):
        call(['python', 'Footwear_testing_window.py'])

    '''
    ### 跳转到手机检测.py文件 ### 
    '''
    def run_script2(self):
        call(['python', 'Mobile_phone_detection_window.py'])




if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
