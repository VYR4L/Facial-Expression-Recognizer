import sys
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QMainWindow, QVBoxLayout, QTextEdit, QComboBox, QFileDialog, QHBoxLayout

class MainWindow(QWidget):
    #Janela principal para a execução da aplicação
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janela Principal da Aplicação")
        
        #O self.caminhoarq aqui ta armazenando certo o endereço do arquivo mas n sei como q funciona pra passar ele pra função do backend
        
        self.caminhoarq = ""

        layout = QVBoxLayout()

        btn = QPushButton("Selecionar Foto")
        btn.clicked.connect(self.SelecionarArquivo)
        btn.setFixedSize(300,100)
        layout.addWidget(btn)

        #Deixei o textbox aqui como placeholder, depois que pegarmos a foto podemos imprimir o resultado dela aqui
        self.textBox = QTextEdit()
        self.textBox.setFixedSize(300,200)

        layout.addWidget(self.textBox)
        
        layout2 = QHBoxLayout()

        #em vez de usar esse label aqui devemos colocar pra usar a camera no lugar e dai setar o tamanho dela e etc.
        label = QLabel("Texto")

        layout2.addWidget(label)
        layout2.addLayout(layout)
        self.setLayout(layout2)

    #Essa é a função de chamar o arquivo, talvez tenhamos q fazer outra função pra chamar ela e passar o caminho do arquivo pra função do backend 
    #e printar o resultado com self.textBox.setText(Resultado)
    def SelecionarArquivo(self):
        arq = 'Imagens (*.png *.jpg *.jpeg *.bmp)'
        nomearq = QFileDialog.getOpenFileName(self, 'Selecione o arquivo', os.getcwd(), arq)
        #O nomearq aqui é uma tupla, o primeiro elemento é o caminho do arquivo, e o segundo ta retornando o tipo de arquivo mas eu ja coloquei o filtro pra td ser imagem
        self.caminhoarq = nomearq[0]


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