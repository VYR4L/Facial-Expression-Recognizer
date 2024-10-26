import sys
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QMainWindow, QVBoxLayout, QTextEdit, QComboBox, QFileDialog, QHBoxLayout

class MainWindow(QWidget):
    #Janela principal para a execução da aplicação
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janela Principal da Aplicação")

        layout = QVBoxLayout()

        btn = QPushButton("Selecionar Foto")
        btn.clicked.connect(self.funcao1)
        btn.setFixedSize(300,100)
        layout.addWidget(btn)

        #Deixei o textbox aqui como placeholder, depois que pegarmos a foto podemos imprimir o resultado dela aqui
        self.textBox = QTextEdit()
        self.textBox.setFixedSize(300,200)
        layout.addWidget(self.textBox)
        
        layout2 = QHBoxLayout()

        #em vez de usar esse label aqui devemos colocar pra usar a camera no mesmo espaço, se pegarmos ela no main etcetc
        label = QLabel("Texto")

        layout2.addWidget(label)
        layout2.addLayout(layout)
        self.setLayout(layout2)

    def funcao1(self):
        opcao = self.options.index(self.combo.currentText())
        if opcao == 1:
            self.SelecionarArquivo()
        else:
            print("Opção inválida")

    def SelecionarArquivo(self):
        arq = 'Imagens (*.png *.jpg *.jpeg *.bmp)'
        nomearq = QFileDialog.getOpenFileName(self, 'Selecione o arquivo', os.getcwd(), arq)
        return nomearq

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet('''
                      QWidget{
                      font-size: 20px;
                      }
                      ''')
    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Closing Window...")