# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camlxz2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def __init__(self):
        self.black_pixmap = QtGui.QPixmap(640, 480)
        self.black_pixmap.fill(QtGui.QColor(0))
        self.turning_pixmap = QtGui.QPixmap("./robocup/turn.png").scaled(72, 70, QtCore.Qt.KeepAspectRatio)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1428, 745)
        self.label_camera = QtWidgets.QLabel(Form)
        self.label_camera.setGeometry(QtCore.QRect(30, 30, 640, 480))
        self.label_camera.setObjectName("label_camera")
        self.button_open_camera = QtWidgets.QPushButton(Form)
        self.button_open_camera.setGeometry(QtCore.QRect(820, 600, 93, 71))
        self.button_open_camera.setObjectName("button_open_camera")
        # self.button_close = QtWidgets.QPushButton(Form)
        # self.button_close.setGeometry(QtCore.QRect(1040, 600, 93, 71))
        # self.button_close.setObjectName("button_close")
        self.label_turning = QtWidgets.QLabel(Form)
        self.label_turning.setGeometry(QtCore.QRect(1040, 600, 72, 70))
        self.label_turning.setObjectName("label_turning")
        self.label_turning.setVisible(False)
        self.label_status = QtWidgets.QLabel(Form)
        self.label_status.setGeometry(QtCore.QRect(720, 440, 431, 32))
        self.label_status.setObjectName("label_status")
        self.table_result = QtWidgets.QTableWidget(Form)
        self.table_result.setGeometry(QtCore.QRect(720, 50, 431, 381))
        self.table_result.setObjectName("table_result")
        self.table_result.setColumnCount(2)
        self.table_result.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table_result.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_result.setHorizontalHeaderItem(1, item)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle("目标识别")
        # self.label_camera.setText("")
        self.label_camera.setPixmap(self.black_pixmap)
        self.button_open_camera.setText("开始")
        # self.button_close.setText("结束")
        self.label_turning.setPixmap(self.turning_pixmap)
        item = self.table_result.horizontalHeaderItem(0)
        item.setText("目标ID")
        item = self.table_result.horizontalHeaderItem(1)
        item.setText("数量")