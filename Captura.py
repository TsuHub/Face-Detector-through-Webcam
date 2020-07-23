'''
REQUISITOS:
Necessário possuir:
1. opencv-contrib-python* (necessário
ser exatamente esta versão, que possui suporte para
captura de rostos) e numpy instalados.
2. Este projeto possui o arquivo xml haar cascade,
responsável pelo treinamento de detecção de objetos
na imagem, o xml "haarcascade_frontalface_default.xml"
precisa estar no mesmo diretório que "Captura.py"


*disponível em:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)

-------------------------------------------------------

EXECUÇÃO:
 e executar no terminal:
$ python Captura.py

O método VideoCapture() pede um argumento inteiro que
representa a câmera (webcam) a ser ativada para a
detecção do objeto, no caso de 1 câmera, entrar com
argumento 0 (zero).
'''
import cv2
hc = 'haarcascade_frontalface_default.xml'
classificador = cv2.CascadeClassifier(hc)    # carrega o arquivo xml haarcascade (hc)
camera = cv2.VideoCapture(0)
sF = 1.5

'''
Um looping infinito, i.e, a detecção ocorre enquanto
o programa estiver rodando e a câmera estiver ligada.
camera.read(): chamada do método para que o objeto
camera possa capturar a imagem.
'''
while (True):
    conectado, img = camera.read()

    # variável responsável por armazenar todas as faces detectadas
    rosto = classificador.detectMultiScale(img, scaleFactor=sF, minSize=(10, 10))
    #scaleFactor = parâmetro que especifica a redução da imagem
    #minSize = objetos menores que 10 x 10 são ignorados

    '''
    cx: coordenada X
    cy: coordenada Y
    Essas coordenadas são referentes ao ponto inicial (superior-esquerdo), então
    o quadrado desenhado na imagem é compensado apenas somando-se a largura e
    altura do rosto.
    '''
    for (cx, cy, largura, altura) in rosto:
        # desenha um retângulo no rosto na imagem (img)
        cv2.rectangle(img, (cx, cy), (cx + largura, cy + altura), (255, 0, 0), 5)
        # (imagem, coordenada da face, cx + largura, cy + altura, cor em BGR, espessura)
        cv2.putText(img, "Rosto", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2)

    print("Conectado: ", conectado, "\nImagem: \n", img)

    cv2.imshow("Face", img)             # argumentos (para mostrar a imagem capturada pela webcam, e variável imagem da captura)
    cv2.waitKey(1)                      # 0 para apresentar os quadros até que se aperte alguma tecla
                                        # e x para um intervalo x, em milisegundos, de um quadro para outro
                                        # sendo x um número inteiro.

camera.release()                        # para liberar a memória após o término
cv2.destroyAllWindows()                 # fecha todas as janelas
