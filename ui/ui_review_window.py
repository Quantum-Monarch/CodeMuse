# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'reviewwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QStatusBar, QWidget)

class Ui_reviewwindow(object):
    def setupUi(self, reviewwindow):
        if not reviewwindow.objectName():
            reviewwindow.setObjectName(u"reviewwindow")
        reviewwindow.resize(800, 600)
        self.centralwidget = QWidget(reviewwindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(0, 0, 771, 491))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 769, 489))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(690, 490, 79, 24))
        reviewwindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(reviewwindow)
        self.statusbar.setObjectName(u"statusbar")
        reviewwindow.setStatusBar(self.statusbar)

        self.retranslateUi(reviewwindow)

        QMetaObject.connectSlotsByName(reviewwindow)
    # setupUi

    def retranslateUi(self, reviewwindow):
        reviewwindow.setWindowTitle(QCoreApplication.translate("reviewwindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("reviewwindow", u"confirm", None))
    # retranslateUi

