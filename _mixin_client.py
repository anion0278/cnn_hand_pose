import socket
import threading
import socketserver
import time

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')

        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def client(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        sock.sendall(bytes(message, 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print("Received: {}".format(response))
    finally:
        sock.close()


class Client():
    
    def connect(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))

    def send(self, message):
        self.sock.sendall(bytes(message, 'ascii'))
        response = str(self.sock.recv(1024), 'ascii')
        print("Received: {}".format(response))

if __name__ == "__main__":
    ip, port = "localhost", 11001

    client = Client()
    client.connect(ip, port)
    client.send("Hello World 1")
    client.send("Hello World 2")
    client.send("Hello World 3")
